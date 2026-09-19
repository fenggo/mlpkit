"""GAP training data — convert ASE trajectories to GAP extended XYZ format."""

import os
import sys
import numpy as np
from ase.io.trajectory import Trajectory


def _write_extxyz(f, atoms, frame_idx, config_type="unknown", write_force=True):
    natom = len(atoms)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # virial (stress × volume)
    has_virial = False
    virial_flat = None
    try:
        stress = atoms.get_stress(voigt=False)
        if stress is not None:
            vol = atoms.get_volume()
            if stress.shape == (6,):
                s = stress
                virial_3x3 = np.array([
                    [s[0], s[5], s[4]],
                    [s[5], s[1], s[3]],
                    [s[4], s[3], s[2]]
                ]) * vol
            elif stress.shape == (3, 3):
                virial_3x3 = stress * vol
            else:
                virial_3x3 = None
            if virial_3x3 is not None:
                virial_flat = virial_3x3.flatten()
                has_virial = True
    except Exception:
        pass

    # properties
    prop_parts = ["species:S:1", "pos:R:3"]
    has_force = forces is not None and write_force
    if has_force:
        prop_parts.append("force:R:3")
    properties = "Properties=" + ":".join(prop_parts)

    # comment line
    lat_flat = cell.flatten()
    lattice_str = " ".join(f"{v:15.8f}" for v in lat_flat)
    lattice_kv = f'Lattice="{lattice_str}"'
    pbc_str = " ".join("T" if p else "F" for p in pbc)
    pbc_kv = f'pbc="{pbc_str}"'
    ct_kv = f"config_type={config_type}"

    comment_parts = [ct_kv, lattice_kv, pbc_kv, properties]
    if energy is not None:
        # CRITICAL: no width format for energy — QUIP parser requires no leading spaces
        comment_parts.insert(1, f"energy={energy:.8f}")
    if has_virial and virial_flat is not None:
        virial_str = " ".join(f"{v:15.8f}" for v in virial_flat)
        comment_parts.insert(1, f'virial="{virial_str}"')

    comment_line = " ".join(comment_parts)

    f.write(f"{natom}\n")
    f.write(f"{comment_line}\n")

    for i in range(natom):
        parts = [f"{symbols[i]:3s}"]
        parts.append(f"{positions[i][0]:15.8f} {positions[i][1]:15.8f} {positions[i][2]:15.8f}")
        if has_force:
            parts.append(f"{forces[i][0]:15.8f} {forces[i][1]:15.8f} {forces[i][2]:15.8f}")
        f.write(" ".join(parts) + "\n")


def gap_convert(ts='', output='train.xyz', config_type='auto',
                skip_no_force=False, no_force=False):
    """Convert ASE .traj files to GAP extended XYZ training format.

    Args:
        ts: space-separated trajectory prefixes (e.g. 'ct4 ct2'). Empty = all .traj
        output: output .xyz file name (default: train.xyz)
        config_type: 'auto' (from filename), 'traj' (use full traj name), or custom string
        skip_no_force: skip frames with no force data
        no_force: omit force data even if present
    """
    cdir = os.getcwd()
    files = os.listdir(cdir)

    if ts:
        prefixes = ts.split()
        trajs = []
        for prefix in prefixes:
            for fil in files:
                if fil.startswith(prefix) and fil.endswith(".traj"):
                    if fil not in trajs:
                        trajs.append(fil)
    else:
        trajs = sorted([f for f in files if f.endswith(".traj")])

    if not trajs:
        print(f"Error: no .traj files found in {cdir}")
        if ts:
            print(f"  filter prefixes: {ts}")
        return

    print(f"Found {len(trajs)} trajectory file(s):")
    for t in trajs:
        print(f"  - {t}")
    print(f"Output: {output}\n")

    total_frames = 0
    skipped = 0

    with open(output, "w") as f:
        for traj_name in trajs:
            images = Trajectory(traj_name)
            n_frames = 0

            for atoms in images:
                if config_type == "auto":
                    ct = traj_name.replace(".traj", "")
                elif config_type == "traj":
                    ct = traj_name
                else:
                    ct = config_type

                forces = atoms.get_forces()
                if skip_no_force and forces is None:
                    skipped += 1
                    continue

                _write_extxyz(f, atoms, n_frames, config_type=ct,
                              write_force=not no_force)
                n_frames += 1

            print(f"  {traj_name}: {n_frames} frames")
            total_frames += n_frames

    print(f"\n✓  Written {total_frames} frames to {output}")
    if skipped:
        print(f"   Skipped {skipped} frames (no forces)")
    print(f"\nNext: gap_fit atoms_filename={output} ...")