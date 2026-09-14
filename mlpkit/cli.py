"""Command-line interface for mlpkit."""

import argparse
import sys
import numpy as np
from mlpkit.core import pred, calc, traj, zmat, fdf, sample,calcdata,gp,fixbroken,add,addall,supercell,update,info,fingerprint,lib,ffield,molinfo
from mlpkit.md2pdf import md2pdf
from mlpkit.deb_bo import dbo

COMMANDS = {
    "pred": (pred, "Predict density/energy using Gaussian Process regression"),
    "calc": (calc, "High-throughput DFT calculation with structure matching"),
    "traj": (traj, "Convert gatheredPOSCARS to ASE trajectory file"),
    "zmat": (zmat, "Convert structure to USPEX Z-matrix format"),
    "fdf":  (fdf,  "Generate SIESTA input files"),
    "sample": (sample, "Sample structures by index to trajectory"),
    "calcdata": (calcdata, "calculate the feature vector of crystal structures"),
    "gp": (gp, "Gaussian process to predict the crystal density"),
    "fixbroken": (fixbroken, "fix broken molecule"),
    "add": (add, "add a structure to data"),
    "addall": (addall, "add a structure to data"),
    "supercell": (supercell, "build a supercell"),
    "update": (update, "update a structure to data"),
    "info": (info, "Print energy and lattice information of a structure"),
    "fingerprint": (fingerprint, "Compute USPEX structural fingerprint (Cython accelerated)"),
    "lib": (lib, "Convert ffield.json to reaxff_nn.lib"),
    "ffield": (ffield, "Convert ffield.json to ReaxFF ffield"),
    "molinfo": (molinfo, "Print molecule atom indices for LAMMPS/COLVARS"),
    "md2pdf": (md2pdf, "Convert Markdown to PDF"),
    "dbo": (dbo, "Plot bond order between two atoms from trajectory"),
    "gmd": (None, "GULP Molecular Dynamics: NVT, optimization, trajectory, plotting"),
    "lmd": (None, "LAMMPS Molecular Dynamics: NVT, NPT, opt, MSST, trajectory, plotting"),
}


def main():
    parser = argparse.ArgumentParser(
        prog="mlpkit",
        description="USPeX Kit — USPEX crystal structure prediction post-processing toolkit",
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    # -- pred --
    p_pred = sub.add_parser("pred", help=COMMANDS["pred"][1])
    p_pred.add_argument("--t", default="Individuals.traj", help="Trajectory file")
    p_pred.add_argument("--g", type=str, default=None, help="geometry structure")
    p_pred.add_argument("--f", type=int, default=1, help="Feature flag (1=8D)")
    p_pred.add_argument("--x", type=int, default=-1, help="index")
    p_pred.add_argument("--den", type=float, default=1.88, help="Density threshold")
    p_pred.add_argument("--ids", default=None, help="Crystal indices (space-separated)")
    p_pred.add_argument("--step", type=int, default=300, help="Optimization steps")
    p_pred.add_argument("--ncpu", type=int, default=8, help="Number of CPUs")
    p_pred.add_argument("--dat", default="data", help="Data directory name")
    p_pred.add_argument("--tolerance", type=float, default=0.001, help="Structure matching tolerance")
    p_pred.add_argument("--c",type=str, default='nn', help="the calculator to be used, aviliable: nn, mtp")

    # -- calc --
    p_calc = sub.add_parser("calc", help=COMMANDS["calc"][1])
    p_calc.add_argument("--t", default="Individuals.traj", help="Trajectory file")
    p_calc.add_argument("--den", type=float, default=1.88, help="Density threshold")
    p_calc.add_argument("--ids", default=None, help="Crystal indices")
    p_calc.add_argument("--step", type=int, default=300, help="MD steps")
    p_calc.add_argument("--ncpu", type=int, default=8, help="Number of CPUs")
    p_calc.add_argument("--dat", default="data", help="Data directory name")
    p_calc.add_argument("--tolerance", type=float, default=0.01, help="Structure matching tolerance")

    # -- traj --
    p_traj = sub.add_parser("traj", help=COMMANDS["traj"][1])
    p_traj.add_argument("--fposcar", default="gatheredPOSCARS", help="Input POSCAR file")

    # -- zmat --
    p_zmat = sub.add_parser("zmat", help=COMMANDS["zmat"][1])
    p_zmat.add_argument("--geo", default="POSCAR", help="Input geometry file")
    p_zmat.add_argument("--i", type=int, default=-1, help="Frame index")

    # -- fdf --
    p_fdf = sub.add_parser("fdf", help=COMMANDS["fdf"][1])
    p_fdf.add_argument("--gen", default="poscar.gen", help="Input .gen file")
    p_fdf.add_argument("--xcf", default="gga", choices=["gga", "vdw"], help="XC functional")
    p_fdf.add_argument("--i", type=int, default=-1, help="Frame index")

    # -- sample --
    p_sample = sub.add_parser("sample", help=COMMANDS["sample"][1])
    p_sample.add_argument("--ind", default="", help="Indices (space-separated)")
    p_sample.add_argument("--t", default=None, help="Trajectory file")
    p_sample.add_argument("--s", type=int, default=None, help="Start frame")
    p_sample.add_argument("--e", type=int, default=None, help="End frame")
    p_sample.add_argument("--i", type=int, default=1, help="Frame interval (default: 1)")
    p_sample.add_argument("--o", default=None, help="Output trajectory file")
    p_sample.add_argument("--f", default="", help="Specific frames (space-separated)")

    # -- calcdata --
    p_calcdata = sub.add_parser("calcdata", help=COMMANDS["calcdata"][1])
    p_calcdata.add_argument("--n",type=int, default=1, help="number cpu tobe used")
    p_calcdata.add_argument("--t", default='structures.traj', help="Trajectory file")
    p_calcdata.add_argument("--step",type=int,  default=1000, help="number of step to used to optimize by MLP")
    p_calcdata.add_argument("--c",type=str, default='nn', help="the calculator to be used, aviliable: nn, mtp")

   # -- gp --  
    p_gp = sub.add_parser("gp", help=COMMANDS["gp"][1])
    p_gp.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_gp.add_argument("--t",type=float, default=0.005, help="structure match tolerance")
    p_gp.add_argument("--step",type=int, default=1000, help="number of step to used to optimize by MLP")
    p_gp.add_argument("--b",type=float, default=1.5, help="energy devate the mean tolerance that the structure is broken")
    p_gp.add_argument("--u",type=float, default=0.03, help="uncertainty of Gaussian Process")
    p_gp.add_argument("--f", type=int,default=1, help="which feature factor to be used")
    p_gp.add_argument("--dft", type=int,default=0, help="whether using active learning and calling DFT")
    p_gp.add_argument("--den", type=float,default=1.82, help="density  criteria to use active learning and calling DFT")
    p_gp.add_argument("--pop", type=int,default=100, help="the population size")
    p_gp.add_argument("--data", default='data', help="which data to be used")
    p_gp.add_argument("--ref", default='results1', help="results file directory")

 # -- fixbroken -- 
    p_fixbroken = sub.add_parser("fixbroken", help=COMMANDS["fixbroken"][1])
    p_fixbroken.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_fixbroken.add_argument("--data", default='data', help="which data to be used")
    p_fixbroken.add_argument("--s", type=float,default=1.2, help="scale factor")
    p_fixbroken.add_argument("--b", type=float,default=1.5, help="energy devate the mean tolerance that the structure is broken")

 # -- add -- 
    p_add = sub.add_parser("add", help=COMMANDS["add"][1])
    p_add.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_add.add_argument("--s", type=int, default=1000, help="the step of mlp geometry optimization")
    p_add.add_argument("--i", type=int, default=-1, help="the index of the Atoms object in trajectory")
    p_add.add_argument("--tolerance",  type=float,default=0.005, help="match tolerance")
    p_add.add_argument("--t", type=str,default='structures.traj', help="trajector file name")
 # -- update -- 
    p_update = sub.add_parser("update", help=COMMANDS["update"][1])
    p_update.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_update.add_argument("--s", type=int, default=1000, help="the step of mlp geometry optimization")
    p_update.add_argument("--tolerance",  type=float,default=0.005, help="match tolerance")
    p_update.add_argument("--t", type=str,default='structures.traj', help="trajector file name")
    p_update.add_argument("--i", default=None, help="Crystal indices (space-separated)")
 # -- addall -- 
    p_addall = sub.add_parser("addall", help=COMMANDS["addall"][1])
    p_addall.add_argument("--n", type=int, default=1, help="number cpu tobe used")
    p_addall.add_argument("--s", type=int, default=1000, help="the step of mlp geometry optimization")
    p_addall.add_argument("--tolerance",  type=float,default=0.005, help="match tolerance")
    p_addall.add_argument("--t", type=str,default='structures.traj', help="trajector file name")
 # -- supercell -- 
    p_supercell = sub.add_parser("supercell", help=COMMANDS["supercell"][1])
    p_supercell.add_argument("--x", type=int, default=1, help="X")
    p_supercell.add_argument("--y", type=int, default=1, help="Y")
    p_supercell.add_argument("--z", type=int, default=1, help="Z")
    p_supercell.add_argument("--t", type=str,default=None, help="trajector file name")
    p_supercell.add_argument("--g", type=str,default=None, help="geometry file name")

 # -- info --
    p_info = sub.add_parser("info", help=COMMANDS["info"][1])
    p_info.add_argument("--gen", default=None, help="Geometry file (e.g. POSCAR, gulp.cif)")
    p_info.add_argument("--traj", default=None, help="Trajectory file name")
    p_info.add_argument("--i", type=int, default=-1, help="Frame index (default: -1)")
    p_info.add_argument("--symmetry", action="store_true", default=True,
                        help="Perform pymatgen symmetry analysis (default: on)")
    p_info.add_argument("--no-symmetry", action="store_false", dest="symmetry",
                        help="Disable symmetry analysis")
    p_info.add_argument("--symprec", type=float, default=0.1,
                        help="Symmetry tolerance for pymatgen (default: 0.1)")

 # -- fingerprint --
    p_fp = sub.add_parser("fingerprint", help=COMMANDS["fingerprint"][1])
    p_fp.add_argument("--g", default=None, help="Geometry structure file (e.g. POSCAR)")
    p_fp.add_argument("--traj", default=None, help="Trajectory file name")
    p_fp.add_argument("--i", type=int, default=-1, help="Frame index (default: -1)")
    p_fp.add_argument("--rmax", type=float, default=12.0, help="Cutoff radius Rmax (Å)")
    p_fp.add_argument("--sigma", type=float, default=0.05, help="Gaussian broadening sigma")
    p_fp.add_argument("--delta", type=float, default=0.08, help="Bin width delta (Å)")
    p_fp.add_argument("--dimension", type=int, default=3, help="Dimension: 3=3D, 0=cluster, 2=2D")
    p_fp.add_argument("--output", default=None, help="Output .npz file (optional)")
    p_fp.add_argument("--intra-map", default=None,
                      help="Intra-molecular distance map (.npy/.npz) for filtering "
                           "intra-molecular pairs. Only zero-shift (basic-cell) pairs "
                           "are filtered; periodic-image pairs are always kept.")
    p_fp.add_argument("--soap", action="store_true",
                      help="Also compute SOAP fingerprint (dscribe). "
                           "Captures local angular environment for better "
                           "discrimination of molecular crystal polymorphs.")
    p_fp.add_argument("--soap-r-cut", type=float, default=6.0,
                      help="SOAP local-environment cutoff radius (Å)")
    p_fp.add_argument("--soap-n-max", type=int, default=8,
                      help="SOAP number of radial basis functions")
    p_fp.add_argument("--soap-l-max", type=int, default=6,
                      help="SOAP maximum angular momentum")

    # -- lib --
    p_lib = sub.add_parser("lib", help=COMMANDS["lib"][1])
    p_lib.add_argument("--json", default="ffield.json", help="Path to ffield.json")
    p_lib.add_argument("--lib", default="reaxff_nn.lib", help="Output lib file name")

    # -- ffield --
    p_ffield = sub.add_parser("ffield", help=COMMANDS["ffield"][1])
    p_ffield.add_argument("--json", default="ffield.json", help="Path to ffield.json")
    p_ffield.add_argument("--ffield", default="ffield", help="Output ffield file name")

    # -- molinfo --
    p_molinfo = sub.add_parser("molinfo", help=COMMANDS["molinfo"][1])
    p_molinfo.add_argument("--g", "--gen", dest="gen", default="data.traj",
                           help="Geometry file (ASE-readable: traj, POSCAR, lammps-data, etc.)")
    p_molinfo.add_argument("--i", type=int, default=-1, help="Frame index (default: -1)")
    p_molinfo.add_argument("--json", dest="jsonfile", default=None,
                           help="Path to ffield.json for bond cutoffs (optional)")
    p_molinfo.add_argument("--quiet", action="store_true", default=False,
                           help="Suppress verbose output (print only atom indices)")

    # -- md2pdf --
    p_md2pdf = sub.add_parser("md2pdf", help=COMMANDS["md2pdf"][1])
    p_md2pdf.add_argument("--i", dest="input", required=True,
                          help="Input file (with or without .md extension)")

    # -- dbo --
    p_dbo = sub.add_parser("dbo", help=COMMANDS["dbo"][1])
    p_dbo.add_argument("--t", "--traj", dest="traj", required=True,
                       help="Trajectory file")
    p_dbo.add_argument("--i", type=int, default=0, help="Index of atom i (default: 0)")
    p_dbo.add_argument("--j", type=int, default=1, help="Index of atom j (default: 1)")
    p_dbo.add_argument("--d", "--delta", dest="delta", type=int, default=1,
                       help="Show Delta value (default: 1)")

    # -- gmd --
    p_gmd = sub.add_parser("gmd", help=COMMANDS["gmd"][1])
    p_gmd.add_argument("--nvt", action="store_true", default=False, help="NVT MD simulation")
    p_gmd.add_argument("--opt", action="store_true", default=False, help="Structure optimization")
    p_gmd.add_argument("--traj", action="store_true", default=False,
                        help="Convert his_3D.arc to trajectory")
    p_gmd.add_argument("--plot", action="store_true", default=False, help="Plot MD results")
    p_gmd.add_argument("--w", action="store_true", default=False, help="Write GULP input only")
    p_gmd.add_argument("--T", type=float, default=350.0, help="Temperature (K)")
    p_gmd.add_argument("--time_step", type=float, default=0.1, help="Time step (fs)")
    p_gmd.add_argument("--step", type=int, default=100, help="Number of steps")
    p_gmd.add_argument("--gen", default="poscar.gen", help="Input geometry file")
    p_gmd.add_argument("--i", type=int, default=-1, help="Frame index")
    p_gmd.add_argument("--mode", default="w", help="Trajectory write mode")
    p_gmd.add_argument("--c", type=int, default=0, help="checkMol flag")
    p_gmd.add_argument("--x", type=int, default=1, help="Supercell X")
    p_gmd.add_argument("--y", type=int, default=1, help="Supercell Y")
    p_gmd.add_argument("--z", type=int, default=1, help="Supercell Z")
    p_gmd.add_argument("--n", type=int, default=1, help="Number of MPI procs for gulp")
    p_gmd.add_argument("--lib", default="reaxff_nn", help="ReaxFF library")
    p_gmd.add_argument("--l", type=int, default=0, help="opt: 0=conv, 1=conp")
    p_gmd.add_argument("--p", type=float, default=0.0, help="Pressure (GPa)")
    p_gmd.add_argument("--inp", default="inp-gulp", help="Input file (traj mode)")
    p_gmd.add_argument("--out", default="out", help="Output file prefix (plot mode)")

    # -- lmd --
    p_lmd = sub.add_parser("lmd", help=COMMANDS["lmd"][1])
    p_lmd.add_argument("--nvt", action="store_true", default=False, help="NVT MD simulation")
    p_lmd.add_argument("--npt", action="store_true", default=False, help="NPT MD simulation")
    p_lmd.add_argument("--opt", action="store_true", default=False, help="Geometry optimization (minimize)")
    p_lmd.add_argument("--msst", action="store_true", default=False, help="MSST shock simulation")
    p_lmd.add_argument("--traj", dest="lmd_traj", action="store_true", default=False,
                       help="Convert lammps.trj to ASE trajectory")
    p_lmd.add_argument("--plot", dest="lmd_plot", action="store_true", default=False,
                       help="Plot LAMMPS thermal output")
    p_lmd.add_argument("--w", action="store_true", default=False, help="Write LAMMPS input only")
    p_lmd.add_argument("--T", type=float, default=350.0, help="Temperature (K)")
    p_lmd.add_argument("--tdump", type=int, default=100, help="Thermostat damping parameter")
    p_lmd.add_argument("--time_step", type=float, default=0.1, help="Time step (fs)")
    p_lmd.add_argument("--step", type=int, default=100, help="Number of steps")
    p_lmd.add_argument("--gen", default="poscar.gen", help="Input geometry file")
    p_lmd.add_argument("--i", type=int, default=-1, help="Frame index")
    p_lmd.add_argument("--model", default="reaxff-nn", help="Model type: reaxff-nn, quip")
    p_lmd.add_argument("--c", type=int, default=0, help="recover flag")
    p_lmd.add_argument("--free", default=" ", help="Free atom indices (space-separated)")
    p_lmd.add_argument("--dump_interval", type=int, default=10, help="Dump interval")
    p_lmd.add_argument("--x", type=int, default=1, help="Supercell X")
    p_lmd.add_argument("--y", type=int, default=1, help="Supercell Y")
    p_lmd.add_argument("--z", type=int, default=1, help="Supercell Z")
    p_lmd.add_argument("--n", type=int, default=1, help="Number of MPI procs")
    p_lmd.add_argument("--lib", default="ffield", help="ReaxFF force field file")
    p_lmd.add_argument("--p", type=float, default=0.0, help="Pressure (NPT mode)")
    p_lmd.add_argument("--r", type=int, default=0, help="Restart flag: 0=data, 1=restart")
    p_lmd.add_argument("--axis", default="z", help="MSST shock direction (x, y, z)")
    p_lmd.add_argument("--v", type=float, default=8.0, help="MSST shock velocity (km/s)")
    p_lmd.add_argument("--q", type=float, default=100.0, help="MSST q parameter")
    p_lmd.add_argument("--s", type=int, default=0, help="Traj start atom")
    p_lmd.add_argument("--e", type=int, default=0, help="Traj end atom")
    p_lmd.add_argument("--inp", default="in.lammps", help="Input file (traj mode)")
    p_lmd.add_argument("--trj", default="lammps.trj", help="trj file (traj mode)")
    p_lmd.add_argument("--out", default="out", help="Output prefix (plot mode)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmd_func = COMMANDS[args.command][0]

    # Map args to function kwargs
    if args.command == "pred":
        cmd_func(t=args.t, g=args.g, f=args.f, den=args.den, ids=args.ids,x=args.x,
                 c=args.c,step=args.step, ncpu=args.ncpu, dat=args.dat, tolerance=args.tolerance)
    elif args.command == "calc":
        cmd_func(t=args.t, den=args.den, ids=args.ids, step=args.step,
                 ncpu=args.ncpu, dat=args.dat, tolerance=args.tolerance)
    elif args.command == "traj":
        cmd_func(fposcar=args.fposcar)
    elif args.command == "zmat":
        cmd_func(geo=args.geo, i=args.i)
    elif args.command == "fdf":
        cmd_func(gen=args.gen, xcf=args.xcf, i=args.i)
    elif args.command == "sample":
        cmd_func(ind=args.ind, t=args.t, s=args.s, e=args.e, i=args.i, o=args.o, f=args.f)
    elif args.command == "calcdata":
        cmd_func(traj=args.t, step=args.step,n=args.n,c=args.c)
    elif args.command == "gp":
        cmd_func(tolerance=args.t,step=args.step,n=args.n,b=args.b,u=args.u,f=args.f,
                 dft=args.dft,pop=args.pop,
                 dat=args.data,ref=args.ref)
    elif args.command == "fixbroken":
        cmd_func(broken=args.b,dat=args.data,scale=args.s,ncpu=args.n)
    elif args.command == "add":
        cmd_func(traj=args.t,tolerance=args.tolerance,step=args.s,i=args.i,ncpu=args.n)
    elif args.command == "addall":
        cmd_func(traj=args.t,tolerance=args.tolerance,step=args.s,ncpu=args.n)
    elif args.command == "supercell":
        cmd_func(traj=args.t,gen=args.g,x=args.x,y=args.y,z=args.z)
    elif args.command == "update":
        cmd_func(traj=args.t,tolerance=args.tolerance,step=args.s,inde=args.i,ncpu=args.n)
    elif args.command == "info":
        cmd_func(gen=args.gen, traj=args.traj, i=args.i,
                 symmetry=args.symmetry, symprec=args.symprec)
    elif args.command == "fingerprint":
        intra_map = None
        if args.intra_map:
            if args.intra_map.endswith(".npz"):
                tmp = np.load(args.intra_map)
                intra_map = tmp[list(tmp.keys())[0]] if len(tmp.files) == 1 else tmp["intra_map"]
            else:
                intra_map = np.load(args.intra_map)
        cmd_func(gen=args.g, traj=args.traj, i=args.i,
                 rmax=args.rmax, sigma=args.sigma, delta=args.delta,
                 dimension=args.dimension, output=args.output,
                 intra_map=intra_map, soap=args.soap,
                 soap_r_cut=args.soap_r_cut, soap_n_max=args.soap_n_max,
                 soap_l_max=args.soap_l_max)
    elif args.command == "lib":
        cmd_func(jsonfile=args.json, libfile=args.lib)
    elif args.command == "ffield":
        cmd_func(jsonfile=args.json, ffieldfile=args.ffield)
    elif args.command == "molinfo":
        cmd_func(gen=args.gen, i=args.i, jsonfile=args.jsonfile,
                 verbose=not args.quiet)
    elif args.command == "md2pdf":
        cmd_func(input=args.input)
    elif args.command == "dbo":
        cmd_func(traj=args.traj, i=args.i, j=args.j, delta=args.delta)
    elif args.command == "gmd":
        from mlpkit.gmd import gmd_dispatch
        gmd_dispatch(args)
    elif args.command == "lmd":
        from mlpkit.lmd import lmd_dispatch
        lmd_dispatch(args)