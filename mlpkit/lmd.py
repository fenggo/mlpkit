"""LAMMPS Molecular Dynamics — LAMMPS MD, optimization, MSST, and analysis."""

from os import system
from ase.io import read
from ase.data import atomic_numbers
from irff.md.lammps import writeLammpsData, writeLammpsIn, get_lammps_thermal, lammpstraj_to_ase


def run_nvt(T=350, tdump=100, timestep=0.1, step=100, gen='poscar.gen', i=-1,
            model='reaxff-nn', c=0, free=' ', dump_interval=10,
            x=1, y=1, z=1, n=1, lib='ffield', thermo_fix=None, r=0):
    atoms = read(gen, index=i) * (x, y, z)
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp = ' '.join(species)
    freeatoms = free.split()
    freeatoms = [int(k) + 1 for k in freeatoms]

    if model == 'quip':
        pair_style = 'quip'
        lib_q = 'Carbon_GAP_20_potential/Carbon_GAP_20.xml ""'
        pair_coeff = '* * {:s} {:d}'.format(lib_q, atomic_numbers[sp])
        units = "metal"
        atom_style = 'atomic'
    else:
        pair_style = 'reaxff control nn yes checkqeq yes'
        pair_coeff = '* * {:s} {:s}'.format(lib, sp)
        units = "real"
        atom_style = 'charge'

    if thermo_fix is None:
        thermo_fix = 'fix   1 all nvt temp {:f} {:f} {:d} '.format(T, T, tdump)

    thermo_style = ('thermo_style  custom step temp epair etotal press vol '
                    'cella cellb cellc cellalpha cellbeta cellgamma '
                    'pxx pyy pzz pxy pxz pyz')

    if r == 0:
        data = 'data.lammps'
        writeLammpsData(atoms, data='data.lammps', specorder=None,
                        masses={'Al': 26.9820, 'C': 12.0000, 'H': 1.0080,
                                'O': 15.9990, 'N': 14.0000, 'F': 18.9980},
                        force_skew=False, velocities=False,
                        units=units, atom_style=atom_style)
    else:
        data = None  # restart mode

    writeLammpsIn(log='lmp.log', timestep=timestep, total=step, restart=None if r == 0 else 'restart',
                  species=species, pair_coeff=pair_coeff, pair_style=pair_style,
                  fix=thermo_fix, freeatoms=freeatoms, natoms=len(atoms),
                  fix_modify=' ', dump_interval=dump_interval, more_commond=' ',
                  thermo_style=thermo_style, units=units, atom_style=atom_style,
                  data=data, restartfile='restart')

    print('\n-  running lammps ...')
    if n == 1:
        system('lammps<in.lammps>out')
    else:
        system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    lammpstraj_to_ase('lammps.trj', inp='in.lammps', recover=c)


def run_npt(T=350, tdump=100, timestep=0.1, step=100, gen='poscar.gen', i=-1,
            model='reaxff-nn', c=0, p=0.0, x=1, y=1, z=1, n=1, lib='ffield',
            free=' ', dump_interval=10, r=0):
    thermo_fix = 'fix   1 all npt temp {:f} {:f} {:d} iso {:f} {:f} {:d}'.format(
        T, T, tdump, p, p, tdump)
    run_nvt(T=T, tdump=tdump, timestep=timestep, step=step, gen=gen, i=i,
            model=model, c=c, free=free, dump_interval=dump_interval,
            x=x, y=y, z=z, n=n, lib=lib, thermo_fix=thermo_fix, r=r)


def run_opt(T=350, timestep=0.1, step=100, gen='poscar.gen', i=-1, model='w',
            c=0, x=1, y=1, z=1, n=1, lib='ffield'):
    atoms = read(gen, index=i) * (x, y, z)
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp = ' '.join(species)

    writeLammpsData(atoms, data='data.lammps', specorder=None,
                    masses={'Al': 26.9820, 'C': 12.0000, 'H': 1.0080,
                            'O': 15.9990, 'N': 14.0000, 'F': 18.9980},
                    force_skew=False, velocities=False,
                    units="real", atom_style='charge')

    writeLammpsIn(log='lmp.log', timestep=timestep, total=step, restart=None,
                  species=species,
                  pair_coeff='* * {:s} {:s}'.format(lib, sp),
                  pair_style='reaxff control nn yes checkqeq yes',
                  fix='minimize\t1e-5 1e-5 2000 2000',
                  fix_modify=' ', more_commond=' ',
                  thermo_style=('thermo_style  custom step temp epair etotal press vol '
                                'cella cellb cellc cellalpha cellbeta cellgamma '
                                'pxx pyy pzz pxy pxz pyz'),
                  data='data.lammps', restartfile='restart')

    print('\n-  running lammps minimize ...')
    if n == 1:
        system('lammps<in.lammps>out')
    else:
        system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    lammpstraj_to_ase('lammps.trj', inp='in.lammps', recover=c)


def run_msst(T=350, timestep=0.1, step=100, gen='poscar.gen', i=-1, model='w',
             c=0, x=1, y=1, z=1, n=1, axis='z', v=8.0, q=100,
             dump_interval=10, free='', lib='ffield', r=1):
    thermo_fix = 'fix msst all msst {:s} {:f} q {:f} mu 3e2 tscale 0.01 '.format(axis, v, q)
    run_nvt(T=T, timestep=timestep, step=step, gen=gen, i=i, model=model, c=c,
            free=free, dump_interval=dump_interval,
            x=x, y=y, z=z, n=n, lib=lib, thermo_fix=thermo_fix, r=r)


def run_traj(inp='in.lammps', s=0, e=0, c=0,trj='lammps.trj',log='lmp.log'):
    atomid = None if e == 0 else (s, e)
    lammpstraj_to_ase(trj, inp=inp, atomid=atomid, recover=c,log=log)


def run_plot(out='out'):
    get_lammps_thermal(logname='lmp.log', supercell=[1, 1, 1])


def run_w(T=350, timestep=0.1, step=100, gen='poscar.gen', i=-1, mode='w',
          c=0, x=1, y=1, z=1, n=1, lib='ffield'):
    atoms = read(gen, index=i) * (x, y, z)
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp = ' '.join(species)

    writeLammpsData(atoms, data='data.lammps', specorder=None,
                    masses={'Al': 26.9820, 'C': 12.0000, 'H': 1.0080,
                            'O': 15.9990, 'N': 14.0000, 'F': 18.9980},
                    force_skew=False, velocities=False,
                    units="real", atom_style='charge')

    writeLammpsIn(log='lmp.log', timestep=timestep, total=step, restart=None,
                  species=species,
                  pair_coeff='* * {:s} {:s}'.format(lib, sp),
                  pair_style='reaxff control nn checkqeq yes',
                  fix='fix   1 all nvt temp 300 300 100.0 ',
                  fix_modify=' ', more_commond=' ',
                  thermo_style=('thermo_style  custom step temp epair etotal press vol '
                                'cella cellb cellc cellalpha cellbeta cellgamma '
                                'pxx pyy pzz pxy pxz pyz'),
                  data='data.lammps', restartfile='restart')
    print('\n-  wrote lammps input files ...')


def lmd_dispatch(args):
    """Dispatch to LAMMPS MD function based on parsed mode flag."""
    if args.nvt:
        run_nvt(T=args.T, tdump=args.tdump, timestep=args.time_step,
                step=args.step, gen=args.gen, i=args.i,
                model=args.model, c=args.c, free=args.free,
                dump_interval=args.dump_interval,
                x=args.x, y=args.y, z=args.z, n=args.n, lib=args.lib, r=args.r)
    elif args.npt:
        run_npt(T=args.T, tdump=args.tdump, timestep=args.time_step,
                step=args.step, gen=args.gen, i=args.i,
                model=args.model, c=args.c, p=args.p,
                x=args.x, y=args.y, z=args.z, n=args.n, lib=args.lib,
                free=args.free, dump_interval=args.dump_interval, r=args.r)
    elif args.opt:
        run_opt(T=args.T, timestep=args.time_step, step=args.step,
                gen=args.gen, i=args.i, model=args.model, c=args.c,
                x=args.x, y=args.y, z=args.z, n=args.n, lib=args.lib)
    elif args.msst:
        run_msst(T=args.T, timestep=args.time_step, step=args.step,
                 gen=args.gen, i=args.i, model=args.model, c=args.c,
                 x=args.x, y=args.y, z=args.z, n=args.n,
                 axis=args.axis, v=args.v, q=args.q,
                 dump_interval=args.dump_interval, free=args.free,
                 lib=args.lib, r=args.r)
    elif args.lmd_traj:
        run_traj(inp=args.inp, s=args.s, e=args.e, c=args.c,trj=args.trj,log=args.log)
    elif args.lmd_plot:
        run_plot(out=args.out)
    elif args.w:
        run_w(T=args.T, timestep=args.time_step, step=args.step,
              gen=args.gen, i=args.i, c=args.c,
              x=args.x, y=args.y, z=args.z, n=args.n, lib=args.lib)
    else:
        print('Error: no mode selected. Use --nvt, --npt, --opt, --msst, --traj, --plot, or --w.')