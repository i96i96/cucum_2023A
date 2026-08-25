# -*- coding: utf-8 -*-
"""
2023 CUMCM A 题问题二 —— 时间保真度一致性预检验
==================================================
检验「cheap 时刻代理」能否代替「全年 60 点平均」作为内层搜索目标。

三个时间保真度：
  t0  noon    单时刻（春分正午）       ~1.8 s/评估
  t1  monthly 12 个「每月 21 日正午」  ~21 s/评估（捕捉季节变化）
  t2  annual  12 月 × 5 时刻 = 60 点   ~139 s/评估（题目规定的精确口径）

对随机采样的布局计算三者，输出两两 Spearman/Pearson 相关：
  * 若 noon/annual 与 monthly/annual 相关都高 → 可用 noon/monthly 做搜索代理，
    仅最终用 annual 校验，内层耗时从 ~30h 降到 ~20–30min。
  * 若 noon/annual 相关低 → 单时刻代理不足，须用 monthly 或更密的时间采样。

运行：python q2_timecheck.py
输出：q2_timecheck.csv
"""

import time
import csv
import numpy as np
from scipy.stats import spearmanr, pearsonr
from q2_layout import (MultiLevelLayout, evaluate_field, evaluate_annual,
                       DIM, M1, M2, DAY_OF_YEAR)

N_SAMPLES = 12
W, H, H_INST = 6.0, 6.0, 4.0


def main():
    layout = MultiLevelLayout()
    rng = np.random.default_rng(7)
    modes = ['noon', 'monthly', 'annual']
    scores = {m: [] for m in modes}

    print('=' * 60)
    print(f'时间保真度一致性预检验（{N_SAMPLES} 组布局）')
    print('=' * 60)
    t0 = time.time()
    for it in range(N_SAMPLES):
        x = np.zeros(DIM)
        x[0] = rng.uniform(-80, 80)
        x[1] = rng.uniform(-80, 80)
        x[2], x[3], x[4] = W, H, H_INST
        x[5] = rng.uniform(100, 130)
        x[6] = int(rng.integers(12, 20))
        x[7] = rng.uniform(12, 20)
        x[8] = rng.uniform(6, 15)
        x[9:9 + M1] = rng.uniform(-0.3, 0.3, size=M1)
        x[9 + M1:9 + M1 + M2] = rng.uniform(-0.4, 0.4, size=M2)

        pos = layout.layout(x, 'L2')
        tx, ty = x[0], x[1]

        scores['noon'].append(evaluate_field(pos, tx, ty, W, H, H_INST, 80, 12.0))
        monthly = [evaluate_field(pos, tx, ty, W, H, H_INST, d, 12.0) for d in DAY_OF_YEAR]
        scores['monthly'].append(float(np.mean(monthly)))
        scores['annual'].append(evaluate_annual(pos, tx, ty, W, H, H_INST))
        print(f'  ... {it + 1}/{N_SAMPLES} 组完成 ({time.time() - t0:.0f}s)')

    print(f'  完成，耗时 {time.time() - t0:.1f}s\n')

    # 相关性
    rows = []
    print('  两两相关（Spearman / Pearson）：')
    for a, b in [('noon', 'annual'), ('monthly', 'annual'), ('noon', 'monthly')]:
        rs, _ = spearmanr(scores[a], scores[b])
        rp, _ = pearsonr(scores[a], scores[b])
        print(f'    {a:>7s} vs {b:<7s}: Spearman {rs:.3f}, Pearson {rp:.3f}')
        rows.append(('spearman', a, b, round(float(rs), 6)))
        rows.append(('pearson', a, b, round(float(rp), 6)))

    # 各模式耗时统计（粗估，按单样本）
    print('\n  各模式单样本均值：')
    for m in modes:
        s = np.array(scores[m])
        print(f'    {m:>7s}: 均值 {s.mean():.4f}')
        rows.append(('mean', m, '', round(float(s.mean()), 6)))

    with open('q2_timecheck.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'mode_a', 'mode_b', 'value'])
        w.writerows(rows)
    print(f'\n结果已写入 q2_timecheck.csv')


if __name__ == '__main__':
    main()
