"""MTP training data — convert ASE trajectories to MTP .cfg format."""

import os
import numpy as np
from ase.io.trajectory import Trajectory

_ATOM_TYPE = {'C': 0, 'O': 1, 'N': 2, 'H': 3}


def _write_cfg(natom, elem, cell, position, force, energy, cfg):
    print('BEGIN_CFG', file=cfg)
    print(' Size', file=cfg)
    print('  {:d}'.format(natom), file=cfg)
    print(' Supercell', file=cfg)
    for c in cell:
        print('  {:f} {:f} {:f}'.format(c[0], c[1], c[2]), file=cfg)
    print(' AtomData: id type  cartes_x  cartes_y  cartes_z   fx   fy  fz', file=cfg)
    for i in range(natom):
        print('{:5d} {:2d} {:10.7} {:10.7} {:10.7}  {:10.7}  {:10.7}  {:10.7}'.format(
            i + 1, _ATOM_TYPE[elem[i]],
            position[i][0], position[i][1], position[i][2],
            force[i][0], force[i][1], force[i][2]), file=cfg)
    print(' Energy', file=cfg)
    print('  {:f}'.format(energy), file=cfg)
    print('END_CFG', file=cfg)
    print('', file=cfg)


def mtp_convert(ts='md', output='train.cfg'):
    """Convert ASE .traj files to MTP .cfg training format.

    Args:
        ts: space-separated trajectory prefixes (e.g. 'ct4 ct2')
        output: output .cfg file name (default: train.cfg)
    """
    cdir = os.getcwd()
    files = os.listdir(cdir)
    prefixes = ts.split()
    trajs = []

    for prefix in prefixes:
        for fil in files:
            if fil.startswith(prefix) and fil.endswith('.traj'):
                trajs.append(fil)

    if not trajs:
        print(f'Error: no .traj files found matching prefixes: {prefixes}')
        return

    cfg = open(output, 'w')
    for traj_name in trajs:
        print(f'  processing {traj_name} ...')
        images = Trajectory(traj_name)
        for atoms in images:
            natom = len(atoms)
            energy = atoms.get_potential_energy()
            force = atoms.get_forces()
            position = atoms.get_positions()
            elem = atoms.get_chemical_symbols()
            cell = atoms.get_cell()
            _write_cfg(natom, elem, cell, position, force, energy, cfg)
    cfg.close()
    print(f'✓  Written to {output} ({len(trajs)} trajectory files)')