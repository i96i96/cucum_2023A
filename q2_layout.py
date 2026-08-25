# -*- coding: utf-8 -*-
"""
2023 CUMCM A 题问题二 —— 多层次布局优化接口（封装层）
========================================================
把预检验阶段散落的「布局生成 + 效率评估」封装为可复用的接口，供正式解题
（多保真度信任域优化）调用。

升级说明（适配 q1_model_fixed 修正版评价器）：
  * 全面移除对旧 q1_model 中遮挡/截断/姿态函数的依赖。
  * 效率评估统一通过 q1_model_fixed.EvaluationConfig + evaluate_field /
    evaluate_one_time 完成，支持可变镜尺寸 w/h、安装高度 H、塔位。
  * 保留布局生成（径向交错 + 径向/角向分区）与多保真度层级结构不变。

设计要点：
  * 4 个嵌套层级 L0/L1/L2/L3 共用同一「径向 M1=3 段 × 角向 M2=4 扇区」几何网格，
    层级差异仅体现在自由参数个数（projection-based 多保真结构）：
        L0  全相等     dr 全局、ds 全局            (9 个自由度)
        L1  径向分段   dr 分 3 段、ds 全局         (12 个自由度)
        L2  径+角分段  dr 分 3 段、ds 分 4 扇区     (16 个自由度)
        L3  逐镜微调   L2 + 每镜偏移               (全自由度，本模块仅留接口钩子)
  * 投影一致性（预检验已验证）：L0 = L1(δ=0) = L2(δ=0,σ=0) = L3(ε=0)。
    这恰好满足多保真度理论（Alexandrov 模型管理 / 空间映射）要求的一致性条件
    f_fine(P(x)) = f_coarse(x)，因此层级切换无需额外修正项。
  * 评估支持可变镜尺寸 w、安装高度 H，并区分「单时刻(noon)」与「年平均(annual)」。

单位：长度 m，角度 rad，效率无量纲。目标方向为「最大化平均光学效率」。
"""

import numpy as np

from q1_model import DAY_OF_YEAR, LOCAL_TIMES, dni, sun_direction
from q1_model_fixed import (
    EvaluationConfig,
    NeighborIndex,
    evaluate_field as _evaluate_field_annual,
    evaluate_one_time,
)

# ======================================================================
# 一、层级与设计向量定义
# ======================================================================
M1 = 3                       # 径向分段数
M2 = 4                       # 角向扇区数
N_GLOBAL = 9                 # 全局变量数
DIM = N_GLOBAL + M1 + M2     # 完整设计向量维度 = 16

LEVELS = ('L0', 'L1', 'L2', 'L3')

# 设计向量分量名（物理量，逐位对应）
VAR_NAMES = (['tower_x', 'tower_y', 'w', 'h', 'H_inst', 'r0', 'K', 'dr', 'ds']
             + [f'delta_{i}' for i in range(M1)]
             + [f'sigma_{i}' for i in range(M2)])

# 变量下/上界（物理单位）。注：K 为整数（生成时四舍五入）。
# 依据 A.md 问题二约束：塔位在 350m 场内；镜边长 2–8m（宽≥高）；
# 安装高度 2–6m；首环半径 ≥100m 禁布区；dr 为「镜宽+5 之上的额外径向间距」，
#   允许为负（交错布局下径向间距可压到 √3/2·(w+5)，即六边形密排）；
# ds 为「镜宽之上的角向安全间距」（≥5 保证相邻镜中心距 ≥ 镜宽+5）。
HEX = 0.5 * np.sqrt(3.0)         # √3/2 ≈ 0.866，六边形密排的径向/角向间距比
VAR_LO = np.array([-150.0, -150.0,   2.0,   2.0,   2.0, 100.0,   5.0,  -2.0,   5.0]
                  + [-0.3] * M1 + [-0.4] * M2, dtype=float)
VAR_HI = np.array([ 150.0,  150.0,   8.0,   8.0,   6.0, 200.0,  50.0,  15.0,  15.0]
                  + [ 0.3] * M1 + [ 0.4] * M2, dtype=float)

# 各层级「激活」的变量索引（未激活者冻结为投影值 0）
ACTIVE = {
    'L0': list(range(N_GLOBAL)),
    'L1': list(range(N_GLOBAL + M1)),
    'L2': list(range(N_GLOBAL + M1 + M2)),
    'L3': list(range(N_GLOBAL + M1 + M2)),
}
INTEGER_VARS = [6]           # K（环数）需取整

FIELD_R = 350.0              # 镜场半径 (m)
FORBID_R = 100.0             # 禁布区半径 (m)


# ======================================================================
# 二、布局生成（径向交错，嵌套设计）
# ======================================================================
def generate_field(tower_x, tower_y, r0, K, dr_seq, ds_by_sector, w):
    """径向交错布局生成。

    dr_seq       : (K,) 第 k 环到第 k+1 环的径向间距
    ds_by_sector : (K, M2) 第 k 环第 s 扇区的周向安全间距
    w            : 镜宽 (m)，用于周向间距约束 (w + ds)
    返回镜位置 (N, 2)。
    """
    positions = []
    r = r0
    M2_ = ds_by_sector.shape[1]
    for k in range(K):
        for s in range(M2_):
            th0 = s * 2 * np.pi / M2_
            th1 = (s + 1) * 2 * np.pi / M2_
            arc = r * (th1 - th0)
            ds = ds_by_sector[k, s]
            n_ks = max(1, int(np.floor(arc / (w + ds))))
            for j in range(n_ks):
                theta = th0 + (j + 0.5 + (k % 2) * 0.5) / n_ks * (th1 - th0)
                x = tower_x + r * np.cos(theta)
                y = tower_y + r * np.sin(theta)
                if x * x + y * y <= FIELD_R ** 2 and \
                   (x - tower_x) ** 2 + (y - tower_y) ** 2 >= FORBID_R ** 2:
                    positions.append((x, y))
        r += dr_seq[k]
    return np.array(positions) if positions else np.zeros((0, 2))


class MultiLevelLayout:
    """多层次布局生成器：从设计向量 x 生成指定层级的布局，并提供投影算子。"""

    def layout(self, x, level='L2'):
        """由完整设计向量 x（物理单位，长度 DIM）生成指定层级布局。

        内置约束处理：
          * 镜宽≥镜高（宽 < 高时交换）；
          * 安装高度 ≥ 镜高/2（保证旋转不触地）；
          * 径向中心距 = 镜宽 + 5 + dr（dr 为额外间距，可为负）；
            交错布局下相邻环对角距 √(a²+(w+ds)²/4) ≥ w+5，故 a ≥ √3/2·(w+5)；
          * 角向安全间距 ds ≥ 5（恒保证同一环相邻镜中心距 ≥ 镜宽+5）。
        """
        x = np.asarray(x, dtype=float)
        tx, ty = x[0], x[1]
        w, h, H = x[2], x[3], x[4]
        if w < h:
            w, h = h, w                       # 约束：宽 ≥ 高
        H = max(H, h / 2.0 + 0.1)             # 约束：旋转不触地
        r0 = x[5]
        K = int(round(x[6]))
        dr = float(x[7])                      # 额外径向间距（可为负，见下钳位）
        ds = max(float(x[8]), 5.0)            # 角向安全间距
        delta = x[9:9 + M1]
        sigma = x[9 + M1:9 + M1 + M2]

        base_dr = w + 5.0 + dr                # 基准径向中心距
        a_min = HEX * (w + 5.0)               # 六边形密排最小径向间距
        base_dr = max(base_dr, a_min)         # L0 亦需满足对角距约束
        seg = np.linspace(0, K, M1 + 1).astype(int)
        dr_seg = np.full(K, base_dr)
        for mm in range(M1):
            dr_seg[seg[mm]:seg[mm + 1]] = base_dr * (1.0 + delta[mm])
        dr_seg = np.maximum(dr_seg, a_min)    # 约束：相邻环对角距 ≥ 镜宽+5

        if level == 'L0':
            dr_seq = np.full(K, base_dr)
            ds_grid = np.full((K, M2), ds)
        elif level == 'L1':
            dr_seq = dr_seg
            ds_grid = np.full((K, M2), ds)
        else:                       # L2 / L3（L3 暂与 L2 同，微调另留接口）
            dr_seq = dr_seg
            ds_grid = np.tile(ds * (1.0 + sigma), (K, 1))
        ds_grid = np.maximum(ds_grid, 5.0)    # 约束：角向中心距 ≥ 镜宽+5

        return generate_field(tx, ty, r0, K, dr_seq, ds_grid, w)

    def project(self, x, to_level):
        """投影算子：把高层级设计向量投影到低层级（冻结高层级自由度）。"""
        y = np.asarray(x, dtype=float).copy()
        if to_level == 'L0':
            y[N_GLOBAL:] = 0.0
        elif to_level == 'L1':
            y[N_GLOBAL + M1:] = 0.0
        return y

    @staticmethod
    def active_dims(level):
        return ACTIVE[level]


# ======================================================================
# 三、效率评估（可变镜尺寸 / 安装高度，复用 q1_model_fixed 修正引擎）
# ======================================================================
def _make_config(tower_x, tower_y, w, h, H_inst):
    """根据 Q2 设计参数构造 q1_model_fixed 的评价配置。"""
    return EvaluationConfig(
        tower_x=float(tower_x),
        tower_y=float(tower_y),
        mirror_width=float(w),
        mirror_height=float(h),
        install_height=float(H_inst),
    )


def evaluate_field(positions, tower_x, tower_y, w, h, H_inst, day=80, st=12.0):
    """单时刻平均光学效率（使用 q1_model_fixed 修正引擎）。"""
    N = len(positions)
    if N == 0:
        return 0.0
    xs = positions[:, 0]
    ys = positions[:, 1]
    sun_vector, alpha = sun_direction(day, st)
    if dni(alpha) <= 0:
        return 0.0

    config = _make_config(tower_x, tower_y, w, h, H_inst)
    index = NeighborIndex(
        xs,
        ys,
        radius=config.neighbor_radius,
        cell_size=config.neighbor_cell_size,
    )
    result = evaluate_one_time(xs, ys, sun_vector, index, config)
    return float(np.mean(result["optical"]))


def evaluate_field_components(positions, tower_x, tower_y, w, h, H_inst, day=80, st=12.0):
    """单时刻全场效率及分量（用于月度表/年度聚合）。

    返回 (eta_opt_mean, eta_cos_mean, eta_sb_mean, eta_trunc_mean, eta_at_mean, DNI, N)。
    """
    N = len(positions)
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    xs = positions[:, 0]
    ys = positions[:, 1]
    sun_vector, alpha = sun_direction(day, st)
    D = float(dni(alpha))
    if D <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, N

    config = _make_config(tower_x, tower_y, w, h, H_inst)
    index = NeighborIndex(
        xs,
        ys,
        radius=config.neighbor_radius,
        cell_size=config.neighbor_cell_size,
    )
    result = evaluate_one_time(xs, ys, sun_vector, index, config)
    return (
        float(np.mean(result["optical"])),
        float(np.mean(result["cosine"])),
        float(np.mean(result["shadow_blocking"])),
        float(np.mean(result["truncation"])),
        float(np.mean(result["atmosphere"])),
        D,
        N,
    )


def compute_annual_metrics(positions, tower_x, tower_y, w, h, H_inst):
    """全年 60 时刻聚合（使用 q1_model_fixed 修正引擎）。

    返回 (monthly, annual)：
      monthly: 12 个月的各效率分量 + 单位面积功率 (kW/m2)
      annual : 年平均光学效率、年平均输出热功率 (kW)、单位镜面面积年平均
               输出热功率 (kW/m2)、总镜面面积 (m2)、镜数 N
    """
    N = len(positions)
    if N == 0:
        monthly = {
            'eta_opt': np.zeros(12),
            'eta_cos': np.zeros(12),
            'eta_sb': np.zeros(12),
            'eta_trunc': np.zeros(12),
            'power_area': np.zeros(12),
        }
        annual = {
            'eta_opt': 0.0,
            'power_total': 0.0,
            'power_area': 0.0,
            'area': 0.0,
            'N': 0,
        }
        return monthly, annual

    config = _make_config(tower_x, tower_y, w, h, H_inst)
    result = _evaluate_field_annual(positions[:, 0], positions[:, 1], config)

    monthly_list = result["monthly"]
    monthly = {
        'eta_opt': np.array([m['平均光学效率'] for m in monthly_list]),
        'eta_cos': np.array([m['平均余弦效率'] for m in monthly_list]),
        'eta_sb': np.array([m['平均阴影遮挡效率'] for m in monthly_list]),
        'eta_trunc': np.array([m['平均截断效率'] for m in monthly_list]),
        'power_area': np.array([m['单位面积镜面平均输出热功率(kW/m2)'] for m in monthly_list]),
    }

    summary = result["summary"]
    area = float(N * w * h)
    annual = {
        'eta_opt': float(summary['annual_optical_efficiency']),
        'power_total': float(summary['annual_output_power_mw'] * 1000.0),
        'power_area': float(summary['annual_unit_area_power_kw_m2']),
        'area': area,
        'N': int(N),
    }
    return monthly, annual


def evaluate_noon_power(positions, tower_x, tower_y, w, h, H_inst):
    """春分正午输出热功率 E_noon (kW)。"""
    eo = evaluate_field(positions, tower_x, tower_y, w, h, H_inst, 80, 12.0)
    sun_vector, alpha = sun_direction(80, 12.0)
    D = float(dni(alpha))
    return D * (w * h) * len(positions) * eo


# ======================================================================
# 四、目标函数封装
# ======================================================================
class FieldObjective:
    """Q2 内层目标：平均光学效率（单位面积功率的代理，DNI 固定时等价）。

    mode='noon'   春分正午单时刻（快，用于调参/预检验）
    mode='annual' 年平均（正式解题用）
    """

    def __init__(self, mode='noon'):
        self.mode = mode
        self.layout = MultiLevelLayout()

    def __call__(self, x, level='L2'):
        x = np.asarray(x, dtype=float)
        pos = self.layout.layout(x, level)
        tx, ty = x[0], x[1]
        w, h, H = x[2], x[3], x[4]
        if len(pos) == 0:
            return 0.0
        if self.mode == 'noon':
            return evaluate_field(pos, tx, ty, w, h, H, 80, 12.0)
        return evaluate_annual(pos, tx, ty, w, h, H)


# 兼容旧接口：FieldObjective('annual') 与 evaluate_annual 同名，这里暴露函数别名
def evaluate_annual(positions, tower_x, tower_y, w, h, H_inst):
    """年平均光学效率（12 月 × 5 时刻，算术平均）。"""
    monthly, annual = compute_annual_metrics(positions, tower_x, tower_y, w, h, H_inst)
    return float(annual['eta_opt'])


# ======================================================================
# 五、归一化辅助（信任域在归一化空间 [0,1]^d 内操作）
# ======================================================================
def to_unit(x):
    return (np.asarray(x, dtype=float) - VAR_LO) / (VAR_HI - VAR_LO)


def from_unit(u):
    return VAR_LO + np.asarray(u, dtype=float) * (VAR_HI - VAR_LO)
