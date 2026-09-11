"""GULP Molecular Dynamics — GULP MD simulation, optimization, and analysis."""

from os import system, popen
import time
from ase.io import read
from irff.md.gulp import write_gulp_in, arctotraj, get_md_results, plot_md, xyztotraj


def run_nvt(T=350.0, time_step=0.1, step=100, gen='poscar.gen', i=-1, mode='w',
            c=0, x=1, y=1, z=1, n=1, lib='reaxff_nn'):
    atoms = read(gen, index=i) * (x, y, z)
    write_gulp_in(atoms, runword='md qiterative conv',
                  T=T, ensemble='nvt', tau_thermostat=0.1,
                  time_step=time_step, tot_step=step, lib=lib)
    print('\n-  running gulp nvt ...')
    if n == 1:
        system('gulp<inp-gulp>gulp.out')
    else:
        system('mpirun -n {:d} gulp<inp-gulp>gulp.out'.format(n))
    xyztotraj('his.xyz', mode=mode, traj='md.traj', checkMol=c, scale=False)


def run_opt(T=350, gen='siesta.traj', step=200, i=-1, l=0, c=0, p=0.0,
            x=1, y=1, z=1, n=1, lib='reaxff_nn'):
    A = read(gen, index=i) * (x, y, z)
    if l == 1 or p > 0.0000001:
        runword = 'opti conp qiterative stre atomic_stress'
    elif l == 0:
        runword = 'opti conv qiterative'
    write_gulp_in(A, runword=runword, T=T, maxcyc=step, pressure=p, lib=lib)
    print('\n-  running gulp optimize ...')
    if n == 1:
        system('gulp<inp-gulp>gulp.out')
    else:
        system('mpirun -n {:d} gulp<inp-gulp>gulp.out'.format(n))
    arctotraj('his_3D.arc', traj='md.traj', checkMol=c)


def run_traj(inp='inp-gulp', c=0):
    arctotraj('his_3D.arc', traj='md.traj', checkMol=c)


def run_plot(out='out'):
    E, Epot, T, P = get_md_results(out=out)
    plot_md(E, Epot, T, P, show=True)


def run_w(T=350, gen='siesta.traj', step=200, i=-1, l=0, c=0,
          x=1, y=1, z=1, lib='reaxff_nn'):
    A = read(gen, index=i) * (x, y, z)
    if l == 0:
        runword = 'opti conv qiterative'
    elif l == 1:
        runword = 'opti conp qiterative stre atomic_stress'
    write_gulp_in(A, runword=runword, T=T, maxcyc=step, lib=lib)
    print('\n-  wrote gulp input file ...')


def gmd_dispatch(args):
    """Dispatch to the correct function based on parsed mode flag."""
    if args.nvt:
        run_nvt(T=args.T, time_step=args.time_step, step=args.step,
                gen=args.gen, i=args.i, mode=args.mode, c=args.c,
                x=args.x, y=args.y, z=args.z, n=args.n, lib=args.lib)
    elif args.opt:
        run_opt(T=args.T, gen=args.gen, step=args.step, i=args.i,
                l=args.l, c=args.c, p=args.p,
                x=args.x, y=args.y, z=args.z, n=args.n, lib=args.lib)
    elif args.traj:
        run_traj(inp=args.inp, c=args.c)
    elif args.plot:
        run_plot(out=args.out)
    elif args.w:
        run_w(T=args.T, gen=args.gen, step=args.step, i=args.i,
              l=args.l, c=args.c, x=args.x, y=args.y, z=args.z, lib=args.lib)
    else:
        print('Error: no mode selected. Use --nvt, --opt, --traj, --plot, or --w.')