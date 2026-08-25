# -*- coding: utf-8 -*-
"""
2023 CUMCM A 题问题二 —— 完整解题脚本（升级约束 + 外层塔位 + 细层级切换）
=====================================================================
升级点：
  1. 吸收塔锁定在 x=0 轴上，外层对 tower_y 做一维搜索；
  2. 内层目标改为「最大化单位面积正午功率 + 60MW 二次约束惩罚」，
     让额定功率缺口成为主导项；
  3. 多保真度优化从 L1 起步，降低 delta_switch，并加入停滞提升规则，
     确保细层级真正参与优化。

运行：python q2_solve.py
产物：
  result2.xlsx            官方答案文件
  q2_table1.csv           表1（月度效率与功率）
  q2_table2.csv           表2（年平均指标）
  q2_table3.csv           表3（设计参数）
  q2_fig_layout.png       最终镜场布局
  q2_fig_monthly.png      月度效率与功率
  q2_fig_convergence.png  内层优化收敛过程
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from q1_model import sun_direction, dni, DAY_OF_YEAR, LOCAL_TIMES
from q2_layout import (MultiLevelLayout, evaluate_field, compute_annual_metrics,
                       DIM, M1, M2, VAR_LO, VAR_HI, LEVELS)
from q2_optimizer import MultiFidelityTR, TRConfig

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

TARGET_KW = 60.0 * 1e3          # 额定功率 60 MW
K_FULL = 50                      # 内层固定环数（足以填满 350m 场，超出被裁剪）
D_NOON = float(dni(sun_direction(80, 12.0)[1]))   # 春分正午 DNI（常数）
TOWER_Y_CANDIDATES = [-120.0, -80.0, -40.0]      # 外层 tower_y 候选（南移优先）


# ==============================================================================
# 内层目标：单位面积正午功率 + 60MW 二次约束惩罚
# ==============================================================================
class Q2InnerObjective:
    """
    max F = p_noon - lam * shortfall^2

    p_noon    : 春分正午单位镜面面积输出热功率 (kW/m2)
                p_noon = D_NOON * eta_opt_noon
    E_noon    : 春分正午总输出热功率 (kW) = D_NOON * A_mirror * N * eta_opt_noon
    shortfall : 相对 60MW-noon 目标的缺口比例
    """

    def __init__(self, layout, beta, lam=3000.0, K=K_FULL, fixed_tower_x=0.0):
        self.layout = layout
        self.beta = beta
        self.lam = lam
        self.K = K
        self.target_noon = TARGET_KW / beta
        self.fixed_tower_x = fixed_tower_x

    def __call__(self, x, level='L2'):
        xk = np.asarray(x, dtype=float).copy()
        xk[0] = self.fixed_tower_x          # 锁定 x=0
        xk[6] = self.K                      # 内层环数固定
        pos = self.layout.layout(xk, level)
        tx, ty = xk[0], xk[1]
        w, h, H = xk[2], xk[3], xk[4]
        if w < h:
            w, h = h, w
        H = max(H, h / 2.0 + 0.1)
        if len(pos) == 0:
            return -self.lam                # 空场：完全不可行
        eo = evaluate_field(pos, tx, ty, w, h, H, 80, 12.0)
        E_noon = D_NOON * (w * h) * len(pos) * eo
        p_noon = D_NOON * eo
        shortfall = max(0.0, (self.target_noon - E_noon) / self.target_noon)
        return p_noon - self.lam * shortfall * shortfall


# ==============================================================================
# 外层：固定 tower_y 后按 K 二分命中 60MW
# ==============================================================================
def noon_power_at_K(shape_x, K, layout):
    xk = shape_x.copy()
    xk[6] = K
    pos = layout.layout(xk, 'L2')
    if len(pos) == 0:
        return 0.0
    w, h, H = xk[2], xk[3], xk[4]
    if w < h:
        w, h = h, w
    H = max(H, h / 2.0 + 0.1)
    eo = evaluate_field(pos, xk[0], xk[1], w, h, H, 80, 12.0)
    return D_NOON * (w * h) * len(pos) * eo


def annual_at_K(shape_x, K, layout):
    xk = shape_x.copy()
    xk[6] = K
    pos = layout.layout(xk, 'L2')
    if len(pos) == 0:
        return None
    w, h, H = xk[2], xk[3], xk[4]
    if w < h:
        w, h = h, w
    H = max(H, h / 2.0 + 0.1)
    monthly, annual = compute_annual_metrics(pos, xk[0], xk[1], w, h, H)
    return K, pos, monthly, annual, xk


def outer_scale_search(shape_x, layout, beta):
    """对给定 tower_y 找最小 K 使 E_annual >= 60MW。"""
    target_noon = TARGET_KW / beta
    lo, hi = 1, K_FULL
    K_noon = K_FULL
    while lo <= hi:
        mid = (lo + hi) // 2
        if noon_power_at_K(shape_x, mid, layout) >= target_noon:
            K_noon = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # 仅在校准后的 K_noon 附近 ±1 做 annual 精确校验，降低计算量
    cands = sorted(set([max(1, K_noon - 1), K_noon, min(K_FULL, K_noon + 1)]))
    best = None
    for K in cands:
        res = annual_at_K(shape_x, K, layout)
        if res is None:
            continue
        K_, pos, monthly, annual, xk = res
        if annual['power_total'] >= TARGET_KW:
            if best is None or annual['power_area'] > best[3]['power_area']:
                best = (K_, pos, monthly, annual, xk)
    if best is None:
        # 即使满场也不满足，则返回满场结果
        return annual_at_K(shape_x, K_FULL, layout)
    return best


# ==============================================================================
# 结果输出
# ==============================================================================
def write_result2_xlsx(xk, pos):
    w, h, H = xk[2], xk[3], xk[4]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['吸收塔x坐标 (m)', '吸收塔y坐标 (m)', '定日镜序号', '定日镜宽度 (m)',
               '定日镜高度 (m)', '定日镜x坐标 (m)', '定日镜y坐标 (m)', '定日镜z坐标 (m)'])
    for i, (mx, my) in enumerate(pos, start=1):
        ws.append([round(xk[0], 4), round(xk[1], 4), i, w, h,
                   round(float(mx), 4), round(float(my), 4), round(H, 4)])
    wb.save('result2.xlsx')
    return len(pos)


def write_tables(monthly, annual):
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                   '7月', '8月', '9月', '10月', '11月', '12月']
    t1 = pd.DataFrame({
        '日期': month_names,
        '平均光学效率': np.round(monthly['eta_opt'], 4),
        '平均余弦效率': np.round(monthly['eta_cos'], 4),
        '平均阴影遮挡效率': np.round(monthly['eta_sb'], 4),
        '平均截断效率': np.round(monthly['eta_trunc'], 4),
        '单位面积镜面平均输出热功率(kW/m2)': np.round(monthly['power_area'], 4),
    })
    t1.to_csv('q2_table1.csv', index=False, encoding='utf-8-sig')

    t2 = pd.DataFrame({
        '指标': ['年平均光学效率', '年平均输出热功率(MW)',
                 '单位镜面面积年平均输出热功率(kW/m2)'],
        '数值': [round(annual['eta_opt'], 4),
                 round(annual['power_total'] / 1e3, 4),
                 round(annual['power_area'], 4)],
    })
    t2.to_csv('q2_table2.csv', index=False, encoding='utf-8-sig')
    return month_names


def write_table3(xk, annual):
    w, h, H = xk[2], xk[3], xk[4]
    t3 = pd.DataFrame({
        '参数': ['吸收塔位置x坐标(m)', '吸收塔位置y坐标(m)', '定日镜宽度(m)',
                 '定日镜高度(m)', '安装高度(m)', '定日镜数目(面)',
                 '定日镜总面积(m2)', '单位镜面面积年平均输出热功率(kW/m2)'],
        '数值': [round(xk[0], 4), round(xk[1], 4), w, h, round(H, 4),
                 annual['N'], round(annual['area'], 2), round(annual['power_area'], 4)],
    })
    t3.to_csv('q2_table3.csv', index=False, encoding='utf-8-sig')


def plot_figures(xk, pos, monthly, annual, month_names, history):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pos[:, 0], pos[:, 1], s=2, c='#2a6f97', alpha=0.7)
    ax.scatter([xk[0]], [xk[1]], s=120, c='r', marker='*', zorder=5, label='吸收塔')
    ax.add_patch(plt.Circle((0, 0), 350, fill=False, ec='k', lw=1.2))
    ax.add_patch(plt.Circle((xk[0], xk[1]), 100, fill=False, ec='r', ls='--', lw=1.2,
                            label='禁布区(100m)'))
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'问题 2：优化后定日镜场布局（{annual["N"]} 面镜）')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig('q2_fig_layout.png', dpi=150)

    xm = np.arange(1, 13)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    a1.plot(xm, monthly['eta_opt'], 'o-', lw=2, label='总光学效率')
    a1.plot(xm, monthly['eta_cos'], 's--', label='余弦效率')
    a1.plot(xm, monthly['eta_sb'], '^--', label='阴影遮挡效率')
    a1.plot(xm, monthly['eta_trunc'], 'v--', label='截断效率')
    a1.set_ylabel('效率')
    a1.set_title('问题 2：12 个月平均光学效率及各分量')
    a1.set_xticks(xm)
    a1.set_xticklabels(month_names, rotation=45)
    a1.grid(alpha=0.3)
    a1.legend()
    a2.bar(xm, monthly['power_area'], color='#2a6f97', alpha=0.85)
    a2.set_xlabel('月份')
    a2.set_ylabel('单位面积平均输出热功率 (kW/m^2)')
    a2.set_xticks(xm)
    a2.set_xticklabels(month_names, rotation=45)
    a2.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig('q2_fig_monthly.png', dpi=150)

    if history:
        f_hist = np.array([h[2] for h in history])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(np.arange(len(f_hist)), f_hist, 'o-', lw=1.2, markersize=3)
        ax.set_xlabel('迭代步数')
        ax.set_ylabel('内层目标 F = p_noon - 惩罚')
        ax.set_title('问题 2：内层多保真度优化收敛过程')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig('q2_fig_convergence.png', dpi=150)


# ==============================================================================
# 主流程
# ==============================================================================
def run_one_tower_y(tower_y, layout, beta, cfg):
    """固定 tower_x=0，对给定 tower_y 运行内层优化 + K 二分。"""
    print(f"\n{'='*60}")
    print(f"[外层] tower_y = {tower_y:.1f} m（吸收塔锁定在 x=0 轴）")
    print('=' * 60)

    x0 = np.array([0.0, tower_y, 6.0, 6.0, 4.0, 120.0, K_FULL, 1.0, 5.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print('[内层] 多保真度信任域优化（L1 起步，tower_x 固定）...')
    obj = Q2InnerObjective(layout, beta, lam=3000.0, K=K_FULL, fixed_tower_x=0.0)
    opt = MultiFidelityTR(obj, cfg, rng=np.random.default_rng(0))
    best_x, best_f, best_level = opt.optimize(x0, start_level='L1')
    shape_x = best_x.copy()
    shape_x[0] = 0.0
    shape_x[6] = K_FULL
    print(f'  内层最优层级={best_level}, 目标 F={best_f:.4f}, 评估次数={opt.evals}')
    print(f'  形状: 塔({shape_x[0]:.1f},{shape_x[1]:.1f}) 镜({shape_x[2]:.2f}x'
          f'{shape_x[3]:.2f}) H={shape_x[4]:.2f} r0={shape_x[5]:.0f} '
          f'dr={shape_x[7]:.1f} ds={shape_x[8]:.1f}')

    print('[外层] 二分 K 使 E_annual 命中 60 MW ...')
    res = outer_scale_search(shape_x, layout, beta)
    if res is None:
        return None, opt.history
    K_star, pos, monthly, annual, xk = res
    P_MW = annual['power_total'] / 1e3
    print(f'  K*={K_star}, E_annual={P_MW:.3f} MW, '
          f'单位面积功率={annual["power_area"]:.4f} kW/m2, N={annual["N"]}')
    return (K_star, pos, monthly, annual, xk, best_level, opt.history), opt.history


def main():
    t0 = time.time()
    layout = MultiLevelLayout()

    # ---------- β 标定（仍以原点基准布局估算 noon->annual 比例） ----------
    x0 = np.array([0.0, 0.0, 6.0, 6.0, 4.0, 120.0, K_FULL, 1.0, 5.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pos0 = layout.layout(x0, 'L2')
    E_noon0 = D_NOON * 36.0 * len(pos0) * evaluate_field(pos0, 0, 0, 6, 6, 4, 80, 12)
    _, ann0 = compute_annual_metrics(pos0, 0, 0, 6, 6, 4)
    beta = ann0['power_total'] / max(E_noon0, 1e-6)
    print(f'[β 标定] β = {beta:.4f}  (E_annual0={ann0["power_total"]/1e3:.2f}MW, '
          f'E_noon0={E_noon0/1e3:.2f}MW, N0={len(pos0)})')

    # ---------- 外层 tower_y 搜索 ----------
    cfg = TRConfig(max_evals=250, max_iter=200, frozen_vars=(0,))
    results = []
    for ty in TOWER_Y_CANDIDATES:
        res, _ = run_one_tower_y(ty, layout, beta, cfg)
        if res is not None:
            results.append(res)

    if not results:
        print('\n错误：所有 tower_y 候选均无可行解。')
        return

    # 优先选择满足 60MW 且单位面积功率最高的；否则选功率最大的
    feasible = [r for r in results if r[3]['power_total'] >= TARGET_KW]
    if feasible:
        best = max(feasible, key=lambda r: r[3]['power_area'])
        print('\n[选择] 在可行解中选择单位面积功率最高者。')
    else:
        best = max(results, key=lambda r: r[3]['power_total'])
        print('\n[选择] 无候选满足 60MW，选择功率最高者。')

    K_star, pos, monthly, annual, xk, best_level, history = best
    P_MW = annual['power_total'] / 1e3

    # ---------- 输出 ----------
    print('\n[输出] 写 result2.xlsx 与图表 ...')
    n_written = write_result2_xlsx(xk, pos)
    month_names = write_tables(monthly, annual)
    write_table3(xk, annual)
    plot_figures(xk, pos, monthly, annual, month_names, history)

    print('\n================ 问题 2 结果 ================')
    print(f'吸收塔位置坐标      : ({xk[0]:.2f}, {xk[1]:.2f}) m')
    print(f'定日镜尺寸(宽x高)   : {xk[2]:.2f} x {xk[3]:.2f} m')
    print(f'安装高度            : {xk[4]:.2f} m')
    print(f'定日镜数目          : {annual["N"]} 面')
    print(f'定日镜总面积        : {annual["area"]:.0f} m2')
    print(f'额定年平均输出热功率: {P_MW:.3f} MW (要求 60 MW)')
    print(f'年平均光学效率      : {annual["eta_opt"]:.4f}')
    print(f'单位镜面面积年平均输出热功率: {annual["power_area"]:.4f} kW/m2')
    print(f'内层最优层级        : {best_level}')
    print(f'result2.xlsx 写入 {n_written} 行')
    print(f'总耗时              : {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
