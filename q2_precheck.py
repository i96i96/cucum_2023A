# -*- coding: utf-8 -*-
"""
2023 CUMCM A 题问题二 —— 建议 6 预检验（驱动脚本）
====================================================
调用 q2_layout 接口做两项检验，并把结果写入 q2_precheck.csv：
  1) 投影一致性：退化参数（δ=0, σ=0, ε=0）下，L0==L1==L2==L3；
  2) 层级相关性：随机采样下，各层级单时刻平均光学效率之间的
     Spearman / Pearson 相关，判断从哪一层级起步合理。

运行：python q2_precheck.py
输出：q2_precheck.csv（长表：metric, level_i, level_j, value）
"""

import time
import csv
import numpy as np
from scipy.stats import spearmanr, pearsonr
from q2_layout import (MultiLevelLayout, evaluate_field, DIM, M1, M2, LEVELS)

EPS_L3 = 0.3              # L3 逐镜微调幅度（预检验专用，接口层暂未内置）
W, H, H_INST = 6.0, 6.0, 4.0   # 预检验阶段固定尺寸（同 Q1）


def _same(a, b):
    if len(a) != len(b):
        return False
    return bool(np.allclose(a, b))


def _evaluate(pos, tx, ty):
    return evaluate_field(pos, tx, ty, W, H, H_INST, 80, 12.0)


def test_projection_consistency(layout):
    """检验 1：投影一致性。返回 [(level_i, level_j, ok), ...]。"""
    print('=' * 60)
    print('检验 1：投影一致性（退化参数下层级布局应完全相同）')
    print('=' * 60)
    x = np.array([0.0, 0.0, W, H, H_INST, 120.0, 16.0, 15.0, 8.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    L0 = layout.layout(x, 'L0')
    L1 = layout.layout(x, 'L1')
    L2 = layout.layout(x, 'L2')
    L3 = L2.copy()                      # ε=0 → L3==L2
    pairs = [('L0', 'L1', L0, L1), ('L1', 'L2', L1, L2), ('L2', 'L3', L2, L3)]
    out = []
    for li, lj, a, b in pairs:
        ok = _same(a, b)
        out.append((li, lj, ok))
        print(f'  {li}=={lj}: {"一致" if ok else "不一致"}  (N={len(a)})')
    return out


def test_correlation(layout, n_samples=60):
    """检验 2：层级相关性。返回 (scores, sizes)。"""
    print('\n' + '=' * 60)
    print(f'检验 2：层级相关性（随机采样 {n_samples} 组，春分正午）')
    print('=' * 60)
    rng = np.random.default_rng(42)
    scores = {lv: [] for lv in LEVELS}
    sizes = {lv: [] for lv in LEVELS}

    t0 = time.time()
    for it in range(n_samples):
        x = np.zeros(DIM)
        x[0] = rng.uniform(-80, 80)
        x[1] = rng.uniform(-80, 80)
        x[2], x[3], x[4] = W, H, H_INST          # 固定尺寸
        x[5] = rng.uniform(100, 130)
        x[6] = int(rng.integers(12, 20))
        x[7] = rng.uniform(12, 20)
        x[8] = rng.uniform(6, 15)
        x[9:9 + M1] = rng.uniform(-0.3, 0.3, size=M1)
        x[9 + M1:9 + M1 + M2] = rng.uniform(-0.4, 0.4, size=M2)

        pos = {'L0': layout.layout(x, 'L0'),
               'L1': layout.layout(x, 'L1'),
               'L2': layout.layout(x, 'L2')}
        if len(pos['L2']):
            pos['L3'] = pos['L2'] + rng.uniform(-EPS_L3, EPS_L3, size=pos['L2'].shape)
        else:
            pos['L3'] = pos['L2']

        for lv in LEVELS:
            scores[lv].append(_evaluate(pos[lv], x[0], x[1]))
            sizes[lv].append(len(pos[lv]))
        if (it + 1) % 10 == 0:
            print(f'  ... {it + 1}/{n_samples} 组完成 ({time.time() - t0:.0f}s)')

    print(f'  完成，耗时 {time.time() - t0:.1f}s')
    return scores, sizes


def write_csv(path, projection, scores, sizes):
    """把全部检测结果写成长表 CSV。"""
    rows = []
    # 投影一致性
    for li, lj, ok in projection:
        rows.append(('projection', li, lj, 1.0 if ok else 0.0))
    # 相关性矩阵
    for metric, func in [('spearman', spearmanr), ('pearson', pearsonr)]:
        for i, li in enumerate(LEVELS):
            for j, lj in enumerate(LEVELS):
                if j < i:
                    continue
                r, _ = func(scores[li], scores[lj])
                rows.append((metric, li, lj, round(float(r), 6)))
    # 汇总统计
    for lv in LEVELS:
        s = np.array(scores[lv])
        rows.append(('mean_efficiency', lv, '', round(float(s.mean()), 6)))
        rows.append(('std_efficiency', lv, '', round(float(s.std()), 6)))
        rows.append(('mean_mirrors', lv, '', round(float(np.mean(sizes[lv])), 3)))
    # 层级递增改善（随机扰动下，非优化，仅供参考）
    for li, lj in [('L1', 'L0'), ('L2', 'L1'), ('L3', 'L2')]:
        d = float(np.mean(np.array(scores[li]) - np.array(scores[lj])))
        rows.append(('improvement', li, lj, round(d, 6)))

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'level_i', 'level_j', 'value'])
        w.writerows(rows)
    print(f'\n结果已写入 {path}（共 {len(rows)} 行）')


def main():
    layout = MultiLevelLayout()
    projection = test_projection_consistency(layout)
    scores, sizes = test_correlation(layout, n_samples=60)
    write_csv('q2_precheck.csv', projection, scores, sizes)


if __name__ == '__main__':
    main()
