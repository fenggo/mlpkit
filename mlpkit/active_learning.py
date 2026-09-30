#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Active Learning — 分块式 ReaxFF-nn 主动学习循环 (mlpkit active).

将原 al_meta.py 改造为 mlpkit 子命令：
  - 工作目录 (META_DIR)   = 当前目录
  - 训练目录 (TRAIN_DIR)  = 当前目录的上一级路径
  - LABEL  由 --label 指定
  - NPROCS 由 --ncpu 指定
  其余变量保持脚本默认值。

流程：
    1. 分块 MD (NVT/NPT + restart 续跑)
    2. mlpkit.critical() 判断失稳
    3. siesta DFT 单点打标签 (lm.py)
    4. 训练 (train.py, 更新 ffield.json)
    5. 力场同步 ffield.json → ffield → META_DIR
    6. 回滚到安全 restart 用新力场继续

依赖：
    - lammps (ReaxFF-nn + COLVARS)
    - mlpkit (Anaconda Python, 含 mlpkit.critical)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
import glob

import numpy as np

# =============================================================
#                  配置 (active() 中重新绑定)                 =
# =============================================================
META_DIR    = os.getcwd()            # metaD 工作目录 (= 当前目录)
TRAIN_DIR   = os.path.abspath('..')  # 训练工作目录 (= 上一级)
LABEL       = 'ct4'
NPROCS      = 12

# Python 环境
ANACONDA_PY = 'python'  # mlpkit 所在 Python

# LAMMPS
LMP         = 'lammps'
MPIRUN      = 'mpirun'

# 数据文件
DATA_FILE   = os.path.join(META_DIR, 'data.lammps')

# ── 元素检测 (从 data.lammps 头注释行) ──
_ELEM_NAME_MAP = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B',
                  6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
                  11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
                  16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
                  21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn',
                  26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn'}

_ELEM_MASS = {1: 1.008, 2: 4.0026, 3: 6.94, 4: 9.0122, 5: 10.81,
              6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 10: 20.180,
              11: 22.990, 12: 24.305, 13: 26.982, 14: 28.085, 15: 30.974,
              16: 32.06, 17: 35.45, 18: 39.948, 19: 39.098, 20: 40.078,
              21: 44.956, 22: 47.867, 23: 50.942, 24: 51.996, 25: 54.938,
              26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546, 30: 65.38}


def detect_elements(data_file=None):
    """从 data.lammps 文件头读取元素映射.

    Returns:
        elements: list of str, 按 type 顺序
        elem_str: 空格分隔字符串, 用于 pair_coeff
    """
    if data_file is None:
        data_file = DATA_FILE
    if not os.path.exists(data_file):
        return None, None

    elem_map = {}

    with open(data_file) as f:
        raw = f.read()

    # 方法 1: #/atom 注释行
    for line in raw.split('\n'):
        s = line.strip()
        if not s.startswith('#/atom '):
            continue
        parts = s.split()
        if len(parts) >= 3:
            try:
                tnum = int(parts[1])
            except ValueError:
                continue
            name = parts[2].lower()
            if len(name) <= 2:
                symbol = name.capitalize()
            else:
                for z, sym in _ELEM_NAME_MAP.items():
                    if sym.lower() == name:
                        symbol = sym
                        break
                else:
                    symbol = name[:2].capitalize()
            elem_map[tnum] = symbol

    if not elem_map:
        print("    ℹ️  从 Masses 段推断元素...")
        in_masses = False
        mass_map = {}
        for line in raw.split('\n'):
            s = line.strip().lower()
            if 'masses' in s:
                in_masses = True
                continue
            if in_masses:
                if not s:
                    continue
                if s.startswith('#') or 'atoms' in s or \
                   'bond' in s or 'angle' in s or 'dihedral' in s or \
                   'pair' in s or 'velocities' in s:
                    break
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        tnum = int(parts[0])
                        mass = float(parts[1])
                    except (ValueError, IndexError):
                        continue
                    mass_map[tnum] = mass

        if mass_map:
            for tnum, mass in sorted(mass_map.items()):
                best_z, best_diff = 0, float('inf')
                for z, sym in _ELEM_NAME_MAP.items():
                    ref_mass = _ELEM_MASS.get(z, 0)
                    if ref_mass == 0:
                        continue
                    diff = abs(mass - ref_mass)
                    if diff < best_diff and diff < 0.6:
                        best_diff = diff
                        best_z = z
                if best_z > 0:
                    elem_map[tnum] = _ELEM_NAME_MAP[best_z]

    if not elem_map:
        print(f"    ⚠️  无法从 {data_file} 检测元素, 请用 --elements 指定")
        return None, None

    elements = [elem_map[i] for i in sorted(elem_map.keys())]
    elem_str = ' '.join(elements)
    print(f"    🔬 检测元素: {elem_str}")
    return elements, elem_str


# =============================================================
# 工具函数
# =============================================================

def run_cmd(cmd, timeout=None, cwd=None, check=True, shell=False, logfile=None):
    """运行命令, 可选将输出写入 log 文件"""
    print(f"\n>>> {' '.join(cmd) if not shell else cmd}")
    t0 = time.time()
    try:
        if logfile:
            with open(logfile, 'w') as fh:
                proc = subprocess.run(cmd, cwd=cwd, shell=shell,
                                      stdout=fh, stderr=subprocess.STDOUT,
                                      text=True, timeout=timeout)
        else:
            proc = subprocess.run(cmd, cwd=cwd, shell=shell,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True, timeout=timeout)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"    [exit {proc.returncode}] ({dt:.0f}s)")
            if not logfile and proc.stdout:
                print(proc.stdout[-800:])
            if check:
                raise RuntimeError(f"命令失败: {cmd[0]} (exit {proc.returncode})")
        else:
            print(f"    [ok] ({dt:.0f}s)")
        return proc
    except subprocess.TimeoutExpired:
        print(f"    [timeout after {timeout}s]")
        return None


def md_healthy(log):
    """检查 MD log: 是否崩溃"""
    if not os.path.exists(log):
        return True
    with open(log, errors='ignore') as f:
        txt = f.read()
    bad = ['ERROR', 'Lost atoms', 'Non-numeric', 'NaN',
           'not enough space', 'MPI_ABORT', 'simulation unstable']
    return not any(b in txt for b in bad)


# =============================================================
# 分块 MD
# =============================================================

def generate_chunk_input(chunk_id, restart_src, nsteps, elements=None, temp=350.0):
    """生成 in.meta_chunk.lammps — 单 chunk 的 LAMMPS 输入."""
    if elements is None:
        elements = ['C', 'H', 'N', 'O']
    elem_str = ' '.join(elements)

    lines = []
    lines.append(f"# Chunk {chunk_id}: {nsteps} steps")
    lines.append("")
    lines.append(f"read_restart    {restart_src}")
    lines.append("")
    lines.append("# ReaxFF-nn")
    lines.append("pair_style      reaxff control nn yes checkqeq yes")
    lines.append(f"pair_coeff      * * ffield {elem_str}")
    lines.append("")
    lines.append("neighbor        2.5  bin")
    lines.append("neigh_modify    every 1 delay 1 check no page 200000")
    lines.append("")
    lines.append("# COLVARS metaD")
    lines.append("")
    lines.append("# NPT")
    lines.append(f"fix             1 all npt temp {temp} {temp} 100 iso 0.0 0.0 100")
    lines.append("fix             Q all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff")
    lines.append("")
    lines.append("thermo_style    custom step temp epair etotal press vol "
                 "cella cellb cellc # f_2")
    lines.append("thermo          1")
    lines.append("")
    dump_file = os.path.join(DUMP_DIR, f"chunk_{chunk_id:04d}.lammpstrj")
    lines.append(f"dump            1 all custom 1 {dump_file} "
                 "id type xu yu zu fx fy fz")
    lines.append("dump_modify     1 sort id")
    lines.append(f"log             meta_npt_chunk_{chunk_id:04d}.log")
    lines.append("")
    lines.append("timestep        0.1")
    lines.append("")
    lines.append(f"run             {nsteps}")
    lines.append("")
    restart_out = f"restart.chunk_{chunk_id:04d}"
    lines.append(f"write_restart   {restart_out}")

    with open(META_IN_CHUNK, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return META_IN_CHUNK, dump_file, restart_out


def run_md_chunk(chunk_id, restart_src, nsteps, elements=None, temp=350.0, timeout_s=36000):
    """运行一个 MD chunk.

    Returns:
        (success: bool, dump_file: str, restart_out: str, log_file: str)
    """
    in_file, dump_file, restart_out = generate_chunk_input(
        chunk_id, restart_src, nsteps, elements, temp)

    log_file = os.path.join(META_DIR, f"meta_npt_chunk_{chunk_id:04d}.log")

    for f in [dump_file, restart_out, log_file]:
        if os.path.exists(f):
            os.remove(f)

    cmd = [MPIRUN, '-np', str(NPROCS), LMP, '-in', in_file]
    print(f"\n{'─'*50}")
    print(f"  Chunk {chunk_id}: {nsteps} 步, "
          f"restart={'初始' if restart_src is None else restart_src}")
    print(f"{'─'*50}")

    try:
        proc = subprocess.run(
            cmd, cwd=META_DIR,
            stdout=open(log_file, 'w'), stderr=subprocess.STDOUT,
            timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"    ⚠️  Chunk {chunk_id}: MD 超时")
        return False, dump_file, restart_out, log_file

    restart_path = os.path.join(META_DIR, restart_out)
    healthy = md_healthy(log_file)
    restart_ok = os.path.exists(restart_path)

    if proc.returncode != 0:
        print(f"    ⚠️  Chunk {chunk_id}: LAMMPS exit {proc.returncode}")
        return False, dump_file, restart_out, log_file

    if not restart_ok:
        print(f"    ⚠️  Chunk {chunk_id}: 未生成 restart 文件")
        return False, dump_file, restart_out, log_file

    if not healthy:
        print(f"    💥 Chunk {chunk_id}: MD 崩溃")
        return False, dump_file, restart_out, log_file

    print(f"    ✅ Chunk {chunk_id}: 完成 → {restart_out}")
    return True, dump_file, restart_out, log_file


# =============================================================
# mlpkit.critical 调用
# =============================================================

CRITICAL_CALL = """\
from mlpkit.core import critical
critical(dump='{dump}', output='{output}', score_threshold={threshold},
         crash_score={crash}, baseline_frames={baseline},
         one_per_run={one_per}, min_persist={min_persist})
"""


def call_mlpkit_critical(dump_file, output='critical.traj',
                         score_threshold=3.0, crash_score=50.0,
                         baseline_frames=10, one_per_run=True,
                         min_persist=3):
    """调用 mlpkit.critical() 判断失稳结构."""
    if not os.path.exists(dump_file):
        print(f"    ❌ dump 不存在: {dump_file}")
        return False, None

    critical_out = os.path.join(META_DIR, output)
    if os.path.exists(critical_out):
        os.remove(critical_out)

    code = CRITICAL_CALL.format(
        dump=dump_file.replace('\\', '\\\\'),
        output=critical_out.replace('\\', '\\\\'),
        threshold=score_threshold,
        crash=crash_score,
        baseline=baseline_frames,
        one_per=str(one_per_run),
        min_persist=min_persist,
    )

    print(f"    🔍 mlpkit.critical: {dump_file}")
    proc = subprocess.run(
        [ANACONDA_PY, '-c', code],
        cwd=META_DIR,
        capture_output=True, text=True, timeout=300)

    if proc.stdout:
        for line in proc.stdout.strip().split('\n'):
            print(f"       {line}")
    if proc.stderr:
        for line in proc.stderr.strip().split('\n'):
            print(f"       [stderr] {line}")

    if os.path.exists(critical_out):
        from ase.io import read
        try:
            frames = read(critical_out, index=':')
            if len(frames) > 0:
                print(f"    ⚠️  发现 {len(frames)} 个失稳帧!")
                return True, critical_out
            else:
                print("    ✅ 无失稳帧")
                return False, None
        except Exception:
            fsize = os.path.getsize(critical_out)
            print(f"    critical.traj 存在但无法解析 (size={fsize})")
            return False, None
    else:
        print("    ✅ 无失稳帧")
        return False, None


# =============================================================
# DFT
# =============================================================

def run_dft(label='cb22', ncpu=None):
    """siesta DFT 单点: 运行 lm.py"""
    if ncpu is None:
        ncpu = NPROCS

    if not os.path.exists(LM_SCRIPT):
        print(f"    ❌ 找不到 {LM_SCRIPT}")
        return False

    cwd = os.getcwd()
    os.chdir(TRAIN_DIR)

    log_path = os.path.join(TRAIN_DIR, 'lm.log')
    proc = subprocess.run(
        [ANACONDA_PY, LM_SCRIPT],
        stdout=open(log_path, 'w'), stderr=subprocess.STDOUT,
        timeout=90000)

    os.chdir(cwd)

    out = os.path.join(TRAIN_DIR, f"{label}.traj")
    if proc.returncode != 0:
        print(f"    ❌ lm.py 失败 (exit {proc.returncode})")
        return False

    print(f"    ✅ DFT 完成: 1 帧带标签 → {label}.traj")
    return True


# =============================================================
# 训练 + 力场同步
# =============================================================

def run_training(epochs, timeout_s=8*3600):
    """训练 ReaxFF-nn"""
    run_cmd([ANACONDA_PY, TRAIN, f'--e={epochs}'],
            cwd=TRAIN_DIR, timeout=timeout_s, check=False)


def sync_ffield():
    """把训练产物 ffield.json 转成文本 ffield 并同步到 MD 工作目录"""
    code = f"""\
import sys
sys.path.insert(0, '/home/xuni/mlpkit')
from mlpkit.core import ffield as mlpkit_ffield
mlpkit_ffield(jsonfile='{FFIELD_JSON}', ffieldfile='{os.path.join(TRAIN_DIR, "ffield")}')
print('ffield.json -> ffield OK')
"""
    proc = subprocess.run(
        [ANACONDA_PY, '-c', code],
        capture_output=True, text=True, timeout=60)
    print(proc.stdout.strip() if proc.stdout else "")
    if proc.stderr:
        print("   ", proc.stderr.strip()[-500:])

    src = os.path.join(TRAIN_DIR, 'ffield')
    if not os.path.exists(src):
        print("    ⚠️ 没有生成 ffield, 力场未同步")
        return False

    shutil.copy(src, FFIELD_META)
    print(f"    🔄 力场同步: {src} → {FFIELD_META}")
    return True


# =============================================================
# 主循环
# =============================================================

def active(label='ct4', ncpu=12, iters=1, epochs=300, chunk_size=1000,
           max_chunks=None, max_md_steps=None, md_timeout=2*3600,
           critical_threshold=3.0, critical_crash=50.0, min_persist=3,
           elements=None, temp=350.0, data_file=None):
    """分块式主动学习循环: chunked MD → critical → DFT → 训练.

    Args:
        label:  训练标签 (用于 {label}.traj 命名)
        ncpu:   MPI 进程数 (NPROCS)
        iters:  主动学习轮数
        epochs: 每轮训练 epoch
        chunk_size: 每 chunk MD 步数
        temp:   MD 温度 (K)
        elements: 元素列表 (空格分隔字符串或 list), 默认自动检测
        data_file: data.lammps 路径
    """
    global META_DIR, TRAIN_DIR, LABEL, NPROCS
    global DATA_FILE, META_IN_CHUNK, COLVARS, FFIELD_META, FFIELD_JSON
    global TRAIN, LM_SCRIPT, RESTART_PATTERN, DUMP_DIR

    # 工作目录 = 当前目录; 训练目录 = 上一级
    META_DIR  = os.getcwd()
    TRAIN_DIR = os.path.abspath('..')
    LABEL     = label
    NPROCS    = ncpu

    DATA_FILE      = data_file or os.path.join(META_DIR, 'data.lammps')
    META_IN_CHUNK  = os.path.join(META_DIR, 'in.meta_chunk.lammps')
    COLVARS        = os.path.join(META_DIR, 'colvars.meta_nvt')
    FFIELD_META    = os.path.join(META_DIR, 'ffield')
    FFIELD_JSON    = os.path.join(TRAIN_DIR, 'ffield.json')
    TRAIN          = os.path.join(TRAIN_DIR, 'train.py')
    LM_SCRIPT      = os.path.join(TRAIN_DIR, 'lm.py')
    RESTART_PATTERN = os.path.join(META_DIR, 'restart.chunk_*')
    DUMP_DIR       = os.path.join(META_DIR, 'chunks')

    print(f"\n{'='*70}")
    print(f"  分块式主动学习")
    print(f"    工作目录 (META):  {META_DIR}")
    print(f"    训练目录 (TRAIN): {TRAIN_DIR}")
    print(f"    LABEL:    {LABEL}")
    print(f"    NPROCS:   {NPROCS}")
    print(f"{'='*70}")

    # ── 元素检测 ──
    elements_list = None
    if elements:
        if isinstance(elements, str):
            elements_list = elements.strip().split()
        else:
            elements_list = list(elements)
        print(f"    🧪 用户指定元素: {' '.join(elements_list)}")
    else:
        elems_detected, _ = detect_elements(DATA_FILE)
        if elems_detected:
            elements_list = elems_detected

    # ── 初始化 ──
    os.makedirs(DUMP_DIR, exist_ok=True)

    # 清理旧的 COLVARS 偏置态
    for f in ['out.colvars.state', 'out.colvars.state.old', 'out.colvars.traj',
              'out.pmf', 'rest.colvars.state']:
        p = os.path.join(META_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"    🧹 清理: {f}")

    for old_rst in glob.glob(RESTART_PATTERN):
        os.remove(old_rst)
        print(f"    🧹 清理旧 restart: {old_rst}")

    init_rst = os.path.join(META_DIR, 'restart.init')
    if os.path.exists(init_rst):
        os.remove(init_rst)

    for old_log in glob.glob(os.path.join(META_DIR, 'meta_npt_chunk_*.log')):
        os.remove(old_log)
    for old_dump in glob.glob(os.path.join(DUMP_DIR, 'chunk_*.lammpstrj')):
        os.remove(old_dump)

    # ── 生成初始 restart ──
    print("\n    🔧 生成 restart.init (零步 run, 从 data.lammps 初始化)...")
    init_in = os.path.join(META_DIR, 'in.init.lammps')
    if elements_list is None:
        elements_list = ['C', 'H', 'N', 'O']
    elem_str = ' '.join(elements_list)
    with open(init_in, 'w') as f:
        f.write(f"""# Generate restart.init (0-step run)
units           real
atom_style      charge

read_data       data.lammps
velocity        all create {temp} {7789}

pair_style      reaxff control nn yes checkqeq yes
pair_coeff      * * ffield {elem_str}
neighbor        2.5  bin
neigh_modify    every 1 delay 1 check no page 200000
fix             1 all npt temp {temp} {temp} 100 iso 0.0 0.0 100
fix             Q all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff
thermo          1
thermo_style    custom step temp epair etotal press vol
timestep        0.1
run             0
write_restart   restart.init
""")

    proc = subprocess.run(
        [MPIRUN, '-np', str(NPROCS), LMP, '-in', init_in],
        cwd=META_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=300)
    init_rst = os.path.join(META_DIR, 'restart.init')
    if proc.returncode != 0 or not os.path.exists(init_rst):
        print(f"    ❌ 生成 restart.init 失败! exit={proc.returncode}")
        sys.exit(1)
    print("    ✅ restart.init 已生成")
    for tmp in [init_in, os.path.join(META_DIR, 'log.lammps')]:
        if os.path.exists(tmp):
            os.remove(tmp)

    restart_src = init_rst
    last_safe_restart = init_rst

    al_iteration = 0
    chunk_id = 0
    total_md_steps = 0
    total_chunks = 0

    print(f"\n{'='*70}")
    print("  主动学习")
    print(f"  每 chunk: {chunk_size} 步")
    print(f"  失稳阈值: score > {critical_threshold}")
    print(f"{'='*70}")

    while True:
        if al_iteration >= iters:
            print(f"\n  🏁 已完成 {al_iteration} 轮主动学习, 结束")
            break

        if max_md_steps and total_md_steps >= max_md_steps:
            print(f"\n  🏁 MD 总步数达到上限 {max_md_steps}, 结束")
            break

        if max_chunks and total_chunks >= max_chunks:
            print(f"\n  🏁 达到最大 chunk 数 {max_chunks}, 结束")
            break

        chunk_id += 1
        total_chunks += 1

        success, dump_file, restart_out, log_file = run_md_chunk(
            chunk_id, restart_src, chunk_size, elements=elements_list,
            temp=temp, timeout_s=md_timeout)
        total_md_steps += chunk_size

        trigger = "success" if success else "CRASH"
        print(f"\n    🔍 mlpkit.critical [{trigger}]: {dump_file}")
        has_critical, critical_traj = call_mlpkit_critical(
            dump_file,
            output=f'critical_{chunk_id:04d}.traj',
            score_threshold=critical_threshold,
            crash_score=critical_crash,
            baseline_frames=min(10, chunk_size // 100),
            min_persist=min_persist)

        if not has_critical:
            if success:
                restart_path = os.path.join(META_DIR, restart_out)
                last_safe_restart = restart_path
                restart_src = restart_path
                print(f"    ✅ 无失稳, 从 {restart_out} 继续")
                continue
            else:
                restart_src = last_safe_restart
                print(f"    ⚠️  Chunk {chunk_id} 崩溃但无失稳帧, "
                      f"回滚到 {restart_src}")
                continue

        al_iteration += 1
        reason = "mlpkit.critical 检测" if success else "MD 崩溃提取"
        print(f"\n{'='*70}")
        print(f"  🔥 主动学习轮 {al_iteration}/{iters}: {reason} → 失稳帧!")
        print(f"{'='*70}")

        samples = os.path.join(TRAIN_DIR, 'samples.traj')
        if critical_traj and os.path.exists(critical_traj):
            shutil.copy(critical_traj, samples)
            print(f"    📦 失稳帧复制: {critical_traj} → {samples}")

        print(f"\n  [{al_iteration}.1] DFT 计算 (siesta)...")
        dft_ok = run_dft(label=LABEL)
        if not dft_ok:
            print("    ❌ DFT 失败, 结束")
            break

        print(f"\n  [{al_iteration}.2] 训练 ReaxFF-nn ({epochs} epochs)...")
        run_training(epochs)

        print(f"\n  [{al_iteration}.3] 力场同步...")
        sync_ffield()

        restart_src = last_safe_restart
        if success:
            print(f"\n  🔄 回滚到安全 restart: {last_safe_restart}")
        else:
            print(f"\n  🔄 (崩溃) 回滚到安全 restart: {last_safe_restart}")
        bad_restart = os.path.join(META_DIR, restart_out)
        if os.path.exists(bad_restart):
            os.remove(bad_restart)

        print("     (使用新力场, 偏置势状态保持)")
        print(f"\n  ✅ 主动学习轮 {al_iteration} 完成. 用新力场继续 MD...")

        if al_iteration >= iters:
            print(f"\n  🏁 已完成 {al_iteration} 轮, 结束")
            break

    print(f"\n{'='*70}")
    print("  主动学习循环结束")
    print(f"    总 chunk 数: {total_chunks}")
    print(f"    总 MD 步数:  {total_md_steps}")
    print(f"    DFT+训练轮数: {al_iteration}/{iters}")
    print(f"{'='*70}")