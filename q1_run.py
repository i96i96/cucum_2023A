# -*- coding: utf-8 -*-
"""
2023 CUMCM A 题问题 1 运行脚本
==============================
读取 附件.xlsx 中 1745 面定日镜坐标，计算 12 月 × 5 时刻的光学效率与
输出热功率，输出表 1 结果（CSV）与可视化图表。

运行：python q1_run.py
产物：
  q1_table1.csv        表 1 结果
  q1_fig_monthly.png   月度效率与功率
  q1_fig_shadow.png    遮挡损失空间分布（春分正午）
  q1_fig_neighbor.png  邻域候选数分布（体现 O(N·k)）
"""

import os
import time
import json
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
except ModuleNotFoundError:
    plt = None

from q1_model import (NeighborIndex, sun_direction, dni, mirror_pose,
                      cosine_efficiency, atmospheric_transmittance,
                      truncation_analytic, truncation_buie, truncation_efficiency,
                      total_efficiency, RECEIVER_CZ, INSTALL_H, MIRROR_W, MIRROR_H,
                      MIRROR_AREA, REFLECTIVITY, DAY_OF_YEAR, LOCAL_TIMES,
                      TRUNC_DSTAR, SUN_SIGMA, RECEIVER_R, RECEIVER_H,
                      _ray_rect_hits, adaptive_eps)

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)


def load_mirrors():
    df = pd.read_excel('附件.xlsx')
    xs = df.iloc[:, 0].astype(float).values
    ys = df.iloc[:, 1].astype(float).values
    return xs, ys


def compute_one_time(xs, ys, idx, s, D, all_pose, seed_base):
    """计算单时刻全场效率。返回每镜的总光学效率 eta_opt (N,) 与各分量。"""
    N = len(xs)
    n_all, u_all, v_all, r_all, d_all = all_pose

    eta_cos = np.zeros(N)
    eta_sb = np.zeros(N)
    eta_trunc = np.zeros(N)
    eta_at = atmospheric_transmittance(d_all)

    c_all = np.stack([xs, ys, np.full(N, INSTALL_H)], axis=1)

    for i in range(N):
        ci = c_all[i]
        ni, ui, vi = n_all[i], u_all[i], v_all[i]
        ri, di = r_all[i], d_all[i]
        # 余弦效率
        eta_cos[i] = cosine_efficiency(s, ni)
        # 阴影遮挡效率（层次化）
        eta_sb[i] = _shadow_blocking(i, ci, ui, vi, s, ri, idx,
                                     xs, ys, c_all, n_all, u_all, v_all)
        # 截断效率（解析—数值混合）
        eta_trunc[i] = truncation_efficiency(di, eta_cos[i], ci, ni, ri,
                                             seed=seed_base + i)

    eta_opt = total_efficiency(eta_cos, eta_at, eta_sb, eta_trunc)
    return eta_opt, eta_cos, eta_sb, eta_trunc, eta_at


def _shadow_blocking(i, center, u, v, s, r, idx, xs, ys, c_all, n_all, u_all, v_all):
    """层次化邻域遮挡检测（L1 网格候选 → L2 投影阈值 → L3 射线求交）。"""
    w, h = MIRROR_W, MIRROR_H
    aa = np.linspace(-w / 2, w / 2, 3)
    bb = np.linspace(-h / 2, h / 2, 3)
    A, B = np.meshgrid(aa, bb)
    points = center[None, :] + A.ravel()[:, None] * u[None, :] + B.ravel()[:, None] * v[None, :]
    M = points.shape[0]
    occ_in = np.zeros(M, dtype=bool)
    occ_re = np.zeros(M, dtype=bool)

    for j in idx.candidates(i):
        cj = c_all[j]
        nj, uj, vj = n_all[j], u_all[j], v_all[j]
        eps = float(adaptive_eps(np.linalg.norm(cj - center)))
        occ_in |= _ray_rect_hits(points, -s, cj, uj, vj, w / 2, h / 2, eps)
        occ_re |= _ray_rect_hits(points, r, cj, uj, vj, w / 2, h / 2, eps)

    return 1.0 - (occ_in | occ_re).sum() / M


def main():
    t0 = time.time()
    xs, ys = load_mirrors()
    N = len(xs)
    print(f'[加载] 定日镜数量 N = {N}')

    idx = NeighborIndex(xs, ys)
    k = np.array([len(idx.candidates(i)) for i in range(N)])
    print(f'[邻域索引] 平均候选数 k = {k.mean():.1f}, 最大 = {k.max()} '
          f'(对比 N^2 = {N * N}, 加速约 {N / k.mean():.0f}x)')

    rec = np.array([0.0, 0.0, RECEIVER_CZ])

    n_months = len(DAY_OF_YEAR)
    n_times = len(LOCAL_TIMES)

    # 月度聚合容器（每月 5 时刻的均值）
    month_eta_opt = np.zeros(n_months)
    month_eta_cos = np.zeros(n_months)
    month_eta_sb = np.zeros(n_months)
    month_eta_trunc = np.zeros(n_months)
    month_power_area = np.zeros(n_months)   # 单位面积平均输出热功率 kW/m2

    annual_power_total = 0.0   # 年平均输出热功率 kW（60 时刻平均）
    annual_eta_opt = 0.0

    # 记录春分正午(3月)的遮挡损失用于空间分布图
    shadow_map = None

    counter = 0
    for m in range(n_months):
        day = DAY_OF_YEAR[m]
        for kk in range(n_times):
            st = LOCAL_TIMES[kk]
            s, alpha = sun_direction(day, st)
            D = dni(alpha)
            if D <= 0:
                continue

            # ---- 预计算全场姿态（向量化）----
            c_all = np.stack([xs, ys, np.full(N, INSTALL_H)], axis=1)
            r_all = rec - c_all
            d_all = np.linalg.norm(r_all, axis=1)
            r_all = r_all / np.maximum(d_all, 1e-9)[:, None]
            n_all = s[None, :] + r_all
            n_all = n_all / np.linalg.norm(n_all, axis=1)[:, None]
            u_all = np.cross(n_all, np.array([0.0, 0.0, 1.0]))
            u_all = u_all / np.maximum(np.linalg.norm(u_all, axis=1), 1e-12)[:, None]
            v_all = np.cross(n_all, u_all)
            all_pose = (n_all, u_all, v_all, r_all, d_all)

            eta_opt, eta_cos, eta_sb, eta_trunc, eta_at = compute_one_time(
                xs, ys, idx, s, D, all_pose, seed_base=m * 10 + kk)

            total_power = D * np.sum(MIRROR_AREA * eta_opt)       # kW
            unit_power = total_power / (N * MIRROR_AREA)          # kW/m2

            month_eta_opt[m] += eta_opt.mean() / n_times
            month_eta_cos[m] += eta_cos.mean() / n_times
            month_eta_sb[m] += eta_sb.mean() / n_times
            month_eta_trunc[m] += eta_trunc.mean() / n_times
            month_power_area[m] += unit_power / n_times

            annual_power_total += total_power / (n_months * n_times)
            annual_eta_opt += eta_opt.mean() / (n_months * n_times)

            # 春分(3月)正午 12:00 记录遮挡损失空间分布
            if m == 2 and st == 12.0:
                shadow_map = 1.0 - eta_sb   # 遮挡损失比例
            counter += 1

    # ---- 输出表 1 ----
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                   '7月', '8月', '9月', '10月', '11月', '12月']
    table = pd.DataFrame({
        '日期': month_names,
        '平均光学效率': np.round(month_eta_opt, 4),
        '平均余弦效率': np.round(month_eta_cos, 4),
        '平均阴影遮挡效率': np.round(month_eta_sb, 4),
        '平均截断效率': np.round(month_eta_trunc, 4),
        '单位面积镜面平均输出热功率(kW/m2)': np.round(month_power_area, 4),
    })
    table.to_csv('q1_table1.csv', index=False, encoding='utf-8-sig')

    annual_unit = annual_power_total / (N * MIRROR_AREA)
    summary = {
        'model': 'original',
        'receiver_center_z_m': float(RECEIVER_CZ),
        'mirror_count': int(N),
        'mirror_area_total_m2': float(N * MIRROR_AREA),
        'annual_optical_efficiency': float(annual_eta_opt),
        'annual_cosine_efficiency': float(month_eta_cos.mean()),
        'annual_shadow_blocking_efficiency': float(month_eta_sb.mean()),
        'annual_truncation_efficiency': float(month_eta_trunc.mean()),
        'annual_output_power_mw': float(annual_power_total / 1e3),
        'annual_unit_area_power_kw_m2': float(annual_unit),
    }
    with open('q1_summary_original.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n================ 问题 1 结果 ================')
    print(table.to_string(index=False))
    print('-----------------------------------------------')
    print(f'年平均光学效率           : {annual_eta_opt:.4f}')
    print(f'年平均输出热功率         : {annual_power_total / 1e3:.3f} MW')
    print(f'单位镜面面积年平均输出热功率 : {annual_unit:.4f} kW/m2')
    print(f'总镜面面积               : {N * MIRROR_AREA:.0f} m2')
    print(f'耗时                     : {time.time() - t0:.1f} s')

    # ---- 绘图 ----
    if plt is not None:
        plot_figures(xs, ys, k, month_names, month_eta_opt, month_eta_cos,
                     month_eta_sb, month_eta_trunc, month_power_area, shadow_map)
    else:
        print('\n[图表] 未安装 matplotlib，跳过绘图；数值结果已完整输出。')


def plot_figures(xs, ys, k, month_names, eta_opt, eta_cos, eta_sb, eta_trunc,
                 power_area, shadow_map):
    xm = np.arange(1, 13)

    # 图 1：月度效率折线
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xm, eta_opt, 'o-', lw=2, label='总光学效率')
    ax.plot(xm, eta_cos, 's--', label='余弦效率')
    ax.plot(xm, eta_sb, '^--', label='阴影遮挡效率')
    ax.plot(xm, eta_trunc, 'v--', label='截断效率')
    ax.set_xlabel('月份'); ax.set_ylabel('效率')
    ax.set_title('问题 1：12 个月平均光学效率及各分量')
    ax.set_xticks(xm); ax.set_xticklabels(month_names, rotation=45)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig('q1_fig_monthly_efficiency.png', dpi=150)

    # 图 2：月度单位面积功率
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(xm, power_area, color='#2a6f97', alpha=0.85)
    ax.set_xlabel('月份'); ax.set_ylabel('单位面积平均输出热功率 (kW/m^2)')
    ax.set_title('问题 1：单位镜面面积年平均输出热功率（每月 21 日）')
    ax.set_xticks(xm); ax.set_xticklabels(month_names, rotation=45)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig('q1_fig_monthly_power.png', dpi=150)

    # 图 3：遮挡损失空间分布（春分正午）
    if shadow_map is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(xs, ys, c=shadow_map, s=8, cmap='hot_r', vmin=0, vmax=0.1)
        ax.add_patch(plt.Circle((0, 0), 350, fill=False, ec='k', lw=1))
        ax.add_patch(plt.Circle((0, 0), 100, fill=False, ec='k', ls='--', lw=1))
        ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title('春分正午：阴影遮挡损失空间分布（颜色越深损失越大）')
        cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label('遮挡损失 (1-eta_sb)')
        fig.tight_layout(); fig.savefig('q1_fig_shadow_map.png', dpi=150)

    # 图 4：邻域候选数分布（体现 O(N·k) 加速）
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(k, bins=40, color='#457b9d', alpha=0.85)
    ax.axvline(k.mean(), color='r', ls='--', label=f'均值 {k.mean():.1f}')
    ax.set_xlabel('每镜邻域候选镜数 k'); ax.set_ylabel('镜数')
    ax.set_title('层次化邻域检测：候选镜数量分布（O(N*k) 中的 k）')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig('q1_fig_neighbor_k.png', dpi=150)

    print('\n[图表] 已保存 q1_fig_*.png')


if __name__ == '__main__':
    main()
