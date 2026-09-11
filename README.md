# mlpkit — machine learning potential toolkit

`mlpkit` 是 USPEX 分子晶体结构预测（计算类型 310）的后处理工具包，提供轨迹转换、特征提取、高斯过程预测、高通量 DFT 筛选、破损分子修复等功能。

## 安装

```bash
git clone https://github.com/FengGo/mlpkit.git
cd  mlpkit
pip install .
```

依赖：`numpy>=1.20`, `scikit-learn>=1.0`, `ase>=3.22`，以及内部库 `irff`（提供 GULP / LAMMPS / SIESTA 接口）。

---

## 命令总览

| 命令 | 功能 |
|------|------|
| `traj` | 将 `gatheredPOSCARS` 转换为 ASE 轨迹文件 |
| `calcdata` | 从轨迹批量计算晶体特征向量 |
| `gp` | 高斯过程预测晶体密度（USPEX 流水线内调用） |
| `pred` | 高斯过程 + 随机森林预测密度和能量 |
| `calc` | 高通量 DFT（SIESTA）计算 + 结构匹配去重 |
| `fixbroken` | 修复破损分子 |
| `add` | 将单个结构添加到特征数据库 |
| `addall` | 将轨迹中所有结构批量添加到特征数据库 |
| `zmat` | 结构坐标转 USPEX Z-matrix 内坐标 |
| `fdf` | 生成 SIESTA 输入文件 |
| `sample` | 按索引或范围从轨迹采样结构 |
| `supercell` | 构建超胞 |
| `update` | 更新数据库中的结构 |
| `info` | 打印结构的能量和晶格信息 |
| `fingerprint` | 计算 USPEX 分子结构指纹（Cython 加速） |
| `lib` | 将 ffield.json 转换为 reaxff_nn.lib |
| `ffield` | 将 ffield.json 转换为 ReaxFF ffield |
| `molinfo` | 打印分子原子索引（LAMMPS/COLVARS 用） |
| `md2pdf` | 将 Markdown 转换为 PDF |
| `dbo` | 绘制轨迹中两原子间键级变化 |
| `gmd` | GULP 分子动力学：NVT、优化、轨迹转换、绘图 |

---

## 1. `traj` — 轨迹转换

将 USPEX 输出文件 `gatheredPOSCARS` 转换为 ASE `.traj` 轨迹文件，同时从 `Individuals` 文件中解析能量信息。

```bash
mlpkit traj [--fposcar FILE]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fposcar` | `gatheredPOSCARS` | 输入 POSCAR 文件路径 |

### 输出

- `Individuals.traj` — ASE 轨迹文件，包含所有结构及能量

### 工作原理

解析 `gatheredPOSCARS` 中的每个结构（以 `EA` 行分隔），写入 `POSCAR` 后用 ASE 读取，再从 `Individuals` 中匹配对应焓值写入 `SinglePointCalculator`。

---

## 2. `calcdata` — 计算特征向量

从 ASE 轨迹中读取结构，通过 MLP（神经网络势）或 MTP（矩张量势）弛豫后，用 GULP 计算能量分解特征，输出到 CSV 数据库。

```bash
mlpkit calcdata [--t TRAJ] [--n NCPU] [--c CALC] [--step STEPS]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `structures.traj` | 输入轨迹文件 |
| `--n` | `1` | 并行 CPU 数 |
| `--c` | `nn` | 计算器类型：`nn`（神经网络反应势）或 `mtp`（MTP 势） |
| `--step` | `1000` | MLP 弛豫步数 |

### 输出

| 文件 | 内容 |
|------|------|
| `feature_mlp.csv` | 10 维特征（GULP 总能 + 能量分解 + 密度） |
| `feature.csv` | 10 维特征（DFT 总能 + GULP 能量分解 + 密度） |
| `structures_mlp.traj` | MLP 弛豫后的结构轨迹 |

### 特征向量 (10 维)

| 维度 | 含义 |
|------|------|
| 0 | 总能 (etot) |
| 1 | 键能 (ebond) |
| 2 | 角能 (eang) |
| 3 | 扭转能 (etor) |
| 4 | 范德华能 (evdw) |
| 5 | 氢键能 C-H-O (ehb_cho) |
| 6 | 氢键能 C-H-N (ehb_chn) |
| 7 | 氢键能 C-H-C (ehb_chc) |
| 8 | 库仑能 (ecoul) |
| 9 | 密度 (density) |

---

## 3. `gp` — 高斯过程预测（USPEX 流水线）

在 USPEX 进化搜索内部调用：对当前结构做 GULP 梯度弛豫 → 提取特征 → GP + RF 预测密度/能量 → 将预测值写回 USPEX 输出格式。

```bash
uspexkit gp [--n NCPU] [--t TOL] [--step STEPS] [--b BROKEN] [--u UNCERT] [--f FEAT] [--data DIR] [--resf DIR]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n` | `1` | 并行 CPU 数 |
| `--t` | `0.005` | 结构匹配容差 |
| `--step` | `1000` | MLP 弛豫步数 |
| `--b` | `1.5` | 破损判断阈值：当前能量偏离均值超过此值视为破损 |
| `--u` | `0.2` | GP 不确定性阈值 |
| `--f` | `1` | 特征标志：`1` = 10 维（含氢键），其他 = 7 维 |
| `--data` | `data` | 训练数据目录名 |
| `--resf` | `results1` | 结果输出目录名 |

### 输出

| 文件 | 内容 |
|------|------|
| `output` | USPEX 格式的能量输出 |
| `optimized.structure` | USPEX 格式的优化结构 |
| `gpr_density.pkl` / `gpr_energy.pkl` | 训练好的 GP 模型 |
| `rfr_density.pkl` | 训练好的随机森林模型 |
| `gp.csv` (在 `results1/` 下) | 预测日志 |

### 工作原理

1. GULP 梯度弛豫当前结构
2. 计算 10 维特征（含 C-H-O / C-H-N / C-H-C 氢键）
3. 从训练数据目录加载 `feature_mlp.csv` / `feature.csv` / `structures.traj`
4. 训练 GP 密度模型 + GP 能量模型 + RF 密度模型（首次训练后缓存为 `.pkl`）
5. 对新结构预测密度和能量，输出到 USPEX 格式
6. 若与最近邻残差 > 10 且 RF 预测偏差大，则回退到缩放修正

---

## 4. `pred` — 预测密度/能量

对指定结构（按索引或 POSCAR 文件）预测密度和能量，使用已训练的 GP + RF 模型。

```bash
mlpkit pred [--t TRAJ] [--g GEO] [--f FEAT] [--den DENSITY] [--ids IDS] [--x INDEX] [--c CALC] [--step STEPS] [--ncpu NCPU] [--dat DIR] [--tolerance TOL]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `Individuals.traj` | 轨迹文件 |
| `--g` | `None` | 几何结构文件（如 `POSCAR`），指定后直接预测该文件 |
| `--f` | `1` | 特征标志：`1` = 10 维（含氢键），其他 = 7 维 |
| `--den` | `1.88` | 密度阈值：只预测密度高于此值的结构 |
| `--ids` | `None` | 晶体索引（空格分隔），如 `"214 215"` |
| `--x` | `-1` | 单个结构索引（`-1` 表示最后一个） |
| `--c` | `nn` | 计算器：`nn`（神经网络势）或 `mtp`（MTP 势） |
| `--step` | `300` | MLP 弛豫步数 |
| `--ncpu` | `8` | 并行 CPU 数 |
| `--dat` | `data` | 训练数据目录名 |
| `--tolerance` | `0.001` | 结构匹配容差 |

### 输出

- `density_predict.log` — 预测日志，每行包含：结构ID、残差、密度（MLP/RF/GP）、GP 不确定性、能量预测

### 使用示例

```bash
# 预测指定索引的结构
mlpkit pred --ids="214 215" --n=24 --dat=data11_44

# 从 POSCAR 文件预测
mlpkit pred --g=POSCAR --n=24 --dat=data11_44
```

---

## 5. `calc` — 高通量 DFT 计算

对符合条件的结构执行 SIESTA DFT 计算，支持结构匹配去重（已计算过的结构自动跳过）。

```bash
mlpkit calc [--t TRAJ] [--den DENSITY] [--ids IDS] [--step STEPS] [--ncpu NCPU] [--dat DIR] [--tolerance TOL]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `Individuals.traj` | 轨迹文件 |
| `--den` | `1.88` | 密度阈值：只计算密度高于此值的结构 |
| `--ids` | `None` | 晶体索引（空格分隔） |
| `--step` | `300` | GULP 弛豫步数 |
| `--ncpu` | `8` | 并行 CPU 数 |
| `--dat` | `data` | 训练数据目录名 |
| `--tolerance` | `0.01` | 结构匹配容差 |

### 输出

- `density.log` — 计算日志
- `{id}/` — 每个结构的独立工作目录，包含 DFT 输入输出
- `{id}/POSCAR.{id}` — 原始结构
- `{id}/POSCAR.{id}_opt` — DFT 优化后结构
- `{id}/id_{id}.traj` — DFT 优化轨迹

### 工作原理

1. 从 `Individuals` 读取结构列表，筛选 `density > den` 且 `fitness < 0` 的结构
2. 对每个结构：GULP 弛豫 → 计算 10 维特征 → 与数据库匹配
3. 若匹配到已知结构（残差 < tolerance）：直接复用已有 DFT 结果
4. 若未匹配：执行 SIESTA DFT 优化（GGA-PBE），并将结果追加到数据库

---

## 6. `fixbroken` — 修复破损分子

检测当前结构是否破损（能量偏离均值超过阈值），若破损则通过扩大晶胞并重新弛豫尝试修复。

```bash
mlpkit fixbroken [--n NCPU] [--data DIR] [--s SCALE] [--b BROKEN]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n` | `1` | 并行 CPU 数 |
| `--data` | `data` | 训练数据目录名 |
| `--s` | `1.2` | 晶胞放大因子（每次迭代乘以该值） |
| `--b` | `1.5` | 破损判断阈值（eV）：当前能量偏离训练数据均值超过此值视为破损 |

### 输出

- `output` — USPEX 格式能量输出
- `optimized.structure` — 优化后结构

### 工作原理

1. GULP 梯度弛豫当前结构
2. 若 `E_mean_train - E_current > broken`：读取分子片段 → 逐步放大晶胞（×1.2，最多 15 次）→ 重新弛豫直到能量恢复正常
3. 若未破损：首次运行时会识别并缓存分子片段（`molecule.pkl`）

---

## 7. `add` — 添加单个结构

将单个结构（DFT 优化后的）追加到特征数据库。

```bash
mlpkit add [--t TRAJ] [--i INDEX] [--s STEPS] [--tolerance TOL] [--n NCPU]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `structures.traj` | 轨迹文件 |
| `--i` | `-1` | 结构索引（`-1` 表示最后一个） |
| `--s` | `1000` | MLP 弛豫步数 |
| `--tolerance` | `0.005` | 结构匹配容差（去重） |
| `--n` | `1` | 并行 CPU 数 |

### 输出

更新 `feature_mlp.csv`、`feature.csv`、`structures_mlp.traj`、`structures.traj`。

---

## 8. `addall` — 批量添加结构

将轨迹中所有结构批量追加到特征数据库。

```bash
mlpkit addall [--t TRAJ] [--s STEPS] [--tolerance TOL] [--n NCPU]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `structures.traj` | 输入轨迹文件 |
| `--s` | `1000` | MLP 弛豫步数 |
| `--tolerance` | `0.005` | 结构匹配容差（去重） |
| `--n` | `1` | 并行 CPU 数 |

---

## 9. `zmat` — Z-matrix 内坐标

将笛卡尔坐标结构转换为 USPEX 格式的 Z-matrix 内坐标文件（`MOL_*`）。

```bash
mlpkit zmat [--geo GEO] [--i INDEX]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--geo` | `POSCAR` | 输入几何文件 |
| `--i` | `-1` | 帧索引（`-1` = 最后一帧） |

### 输出

- USPEX 格式的 `MOL_*` 内坐标文件

### 使用示例

```bash
mlpkit zmat --g=POSCAR
```

---

## 10. `fdf` — 生成 SIESTA 输入

从结构文件生成 SIESTA DFT 的 `.fdf` 输入文件。

```bash
mlpkit fdf [--gen GEN] [--xcf XCF] [--i INDEX]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gen` | `poscar.gen` | 输入 `.gen` 格式结构文件 |
| `--xcf` | `gga` | 交换关联泛函：`gga`（GGA-PBE）或 `vdw`（VDW-DRSLL） |
| `--i` | `-1` | 帧索引 |

---

## 11. `sample` — 采样结构

按索引、帧号列表或范围从轨迹中采样结构。

```bash
mlpkit sample [--ind INDICES] [--t TRAJ] [--s START] [--e END] [--i INTERVAL] [--o OUTPUT] [--f FRAMES]
```

### 三种模式（优先级从高到低）

| 模式 | 参数 | 说明 |
|------|------|------|
| 显式帧号 | `--f` | 空格分隔的帧索引列表 |
| Legacy 索引 | `--ind` | 空格分隔的索引列表（兼容旧用法） |
| 范围模式 | `--s / --e / --i` | 按起始帧→结束帧，每隔 interval 帧采样 |

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--t` | `None` | 输入轨迹文件 |
| `--ind` | `""` | 结构索引（空格分隔），如 `"0 5 12"` |
| `--s` | `None` | 起始帧索引 |
| `--e` | `None` | 结束帧索引（含） |
| `--i` | `1` | 采样间隔 |
| `--o` | `None` | 输出轨迹文件路径（默认 `samples.traj`） |
| `--f` | `""` | 显式帧号（空格分隔），如 `"1 3 10 15"` |

### 输出

- `samples.traj`（或通过 `--o` 指定）— 采样后的结构轨迹

### 使用示例

```bash
# 范围模式：帧 1~20，每隔 2 帧采样
mlpkit sample --s=1 --e=20 --i=2 --t=md.traj --o=samples.traj

# 显式帧号：提取指定帧
mlpkit sample --t=md.traj --f="0 5 10 15" --o=selected.traj

# Legacy 索引模式（兼容旧用法）
mlpkit sample --ind="0 1 2" --t=structures.traj
```

---

## 12. `supercell` — 构建超胞

从结构或轨迹构建超胞。

```bash
mlpkit supercell [--x NX] [--y NY] [--z NZ] [--t TRAJ] [--g GEO]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--x` | `1` | X 方向倍数 |
| `--y` | `1` | Y 方向倍数 |
| `--z` | `1` | Z 方向倍数 |
| `--t` | `None` | 轨迹文件（取最后一帧） |
| `--g` | `None` | 几何文件（如 `POSCAR`） |

### 输出

- 若指定 `--g`：输出 `POSCAR.supercell_{x}_{y}_{z}`
- 若指定 `--t`：输出 `{traj_name}_{x}{y}{z}.traj`，能量按体积比例缩放

---

## 13. `fingerprint` - 分子结构指纹

使用 Cython 加速计算 USPEX 分子结构指纹（`makeMatrices` + `fingerprint_calc`），输出 `order`、`fing`、`atom_fing` 三个指纹数组。

```bash
mlpkit fingerprint [--g GEO] [--traj TRAJ] [--i I] [--rmax RMAX] [--sigma SIGMA] [--delta DELTA] [--dimension DIM] [--output OUTPUT]
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--g` | `None` | 几何结构文件（如 `POSCAR`、`gulp.cif`），用 ASE 读取 |
| `--traj` | `None` | 轨迹文件（与 `--g` 二选一） |
| `--i` | `-1` | 轨迹帧索引（`-1` = 最后一帧） |
| `--rmax` | `12.0` | 近邻搜索截断半径 Rmax（Å） |
| `--sigma` | `0.05` | 高斯展宽 σ |
| `--delta` | `0.08` | 距离分箱宽度 δ（Å） |
| `--dimension` | `3` | 维度：`3` = 3D 晶体，`0` = 团簇，`2` = 2D |
| `--output` | `None` | 输出 `.npz` 文件路径（可选，不指定则仅打印摘要） |

### 输出

- 终端打印指纹统计信息（形状、最值、均值、耗时）
- 若指定 `--output`：保存 `.npz` 文件，包含 `order`、`fing`、`atom_fing`、`V`、`n_pairs` 等数组

### 使用示例

```bash
# 从 POSCAR 计算指纹
mlpkit fingerprint --g=POSCAR

# 自定义参数并保存结果
mlpkit fingerprint --g=POSCAR --rmax=10.0 --sigma=0.07 --output=fp.npz

# 从轨迹文件计算
mlpkit fingerprint --traj=Individuals.traj --i=-1
```

### 工作原理

1. 用 ASE 读取结构文件，提取晶格、分数坐标、元素种类和原子数
2. 调用 Cython 加速模块 `uspex_fast_core.compute_all`：
   - `build_distance_matrix`：构建近邻距离矩阵（替代 Octave `makeMatrices.m`）
   - `fingerprint_calc`：计算 erf 展宽距离直方图指纹（替代 Octave `fingerprint_calc.m`）
3. 输出三个指纹数组：
   - `order` (N,)：每个原子的结构序参量（√(Σ weight·δ·atom_fing²/V^(1/3))）
   - `fing` (S², numBins)：全局指纹矩阵（S = 元素种类数）
   - `atom_fing` (N, S, numBins)：原子级指纹

---

## 14. `gmd` — GULP 分子动力学

GULP MD 仿真、结构优化、轨迹转换和结果绘图。

```bash
mlpkit gmd --nvt|--opt|--traj|--plot|--w [选项...]
```

### 五种模式（互斥）

| 模式 | 说明 |
|------|------|
| `--nvt` | NVT 系综分子动力学仿真 |
| `--opt` | 结构优化（GULP gradient + Hessian） |
| `--traj` | 将 GULP 输出 `his_3D.arc` 转换为 ASE 轨迹 |
| `--plot` | 绘制 MD 结果（能量/温度/压力 vs 步数） |
| `--w` | 仅写入 GULP 输入文件，不执行计算 |

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--T` | `350.0` | 温度 (K) |
| `--step` | `100` | 步数（MD 步数或优化最大循环） |
| `--gen` | `poscar.gen` | 输入结构文件（ASE 可读） |
| `--i` | `-1` | 帧索引 |
| `--x / --y / --z` | `1` | 超胞倍数 |
| `--n` | `1` | MPI 进程数（并行 gulp） |
| `--lib` | `reaxff_nn` | ReaxFF 力场库名称 |
| `--c` | `0` | checkMol 标志 |

### NVT 模式额外参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--time_step` | `0.1` | 时间步长 (fs) |
| `--mode` | `w` | 轨迹写入模式 |

### Opt 模式额外参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--l` | `0` | 优化类型：`0` = conv（恒容），`1` = conp（恒压） |
| `--p` | `0.0` | 外部压力 (GPa)，> 0 自动启用 conp |

### Traj / Plot 模式参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--inp` | `inp-gulp` | GULP 输入文件（traj 模式） |
| `--out` | `out` | 输出文件前缀（plot 模式） |

### 使用示例

```bash
# NVT MD：500 K，1000 步，步长 0.1 fs
mlpkit gmd --nvt --T=500 --step=1000 --time_step=0.1 --gen=poscar.gen

# NVT 4 核并行
mlpkit gmd --nvt --n=4 --gen=poscar.gen

# 恒容优化
mlpkit gmd --opt --gen=siesta.traj --step=200

# 恒压优化（0.5 GPa）
mlpkit gmd --opt --l=1 --p=0.5 --gen=siesta.traj

# 仅写入输入文件（不运行）
mlpkit gmd --w --gen=my.gen --lib=reaxff_nn

# 转换 arc 到 traj
mlpkit gmd --traj --c=1

# 绘制 MD 结果
mlpkit gmd --plot --out=out
```

### 输出

| 模式 | 输出 |
|------|------|
| `--nvt` | `gulp.out` 输出 + `md.traj` 轨迹 (xyz → traj) |
| `--opt` | `gulp.out` + `md.traj` (his_3D.arc → traj) |
| `--w` | `inp-gulp`（GULP 输入文件） |
| `--traj` | `md.traj`（从 his_3D.arc 转换） |
| `--plot` | 交互式图表（能量/温度/压力） |

---

## 数据格式说明

### 指纹计算输出 (`.npz`)

`fingerprint --output` 生成的 `.npz` 文件包含以下数组：

| 键 | 形状 | 说明 |
|------|------|------|
| `order` | (N,) | 每个原子的结构序参量 |
| `fing` | (S², numBins) | 全局指纹矩阵（S = 元素种类数） |
| `atom_fing` | (N, S, numBins) | 原子级指纹 |
| `V` | 标量 | 晶胞体积 |
| `n_pairs` | 标量 | 近邻原子对数 |
| `numIons` | (S,) | 各元素原子数 |
| `atomType` | (S,) | 各元素原子序数 |
| `rmax` / `sigma` / `delta` / `dimension` | 标量 | 计算参数 |

### `feature_mlp.csv` (GULP 能量特征)

```
, etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density
0, -123.45, -56.78, ...
```

第一列为结构索引，后续为 10 维特征向量。

### `feature.csv` (DFT 能量 + GULP 分解)

```
, etot, ebond, eang, etor, evdw, ehb_cho, ehb_chn, ehb_chc, ecoul, density
0, -234.56, -56.78, ...
```

第一列为 DFT 总能（来自 `SinglePointCalculator`），后续为 GULP 分解能量。

### GP 模型文件

- `gpr_density.pkl` — 高斯过程密度模型
- `gpr_energy.pkl` — 高斯过程能量模型
- `rfr_density.pkl` — 随机森林密度模型

核函数：`0.00581² · DotProduct(σ₀=0.412) + 0.35² · Matern(ν=2.5, ARD) + WhiteKernel`

---

## 典型工作流

### 1. 构建训练数据库

```bash
# 从 DFT 结果生成特征数据库
mlpkit calcdata --t=structures.traj --n=24

# 或手动添加结构
mlpkit add --t=structures.traj --i=-1 --n=24
```

### 2. 进化搜索中预测

在 USPEX 的 `command` 中配置：
```bash
mlpkit gp --n=24 --data=data11_44 --resf=results1
```

### 3. 高通量 DFT 筛选

```bash
mlpkit calc --n=24 --den=1.88 --dat=data11_44
```

### 4. 破损分子修复

```bash
mlpkit fixbroken --n=24 --data=data11_44 --b=1.5
```
