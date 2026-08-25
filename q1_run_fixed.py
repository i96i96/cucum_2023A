# -*- coding: utf-8 -*-
"""运行问题一修正版评价器并保存可审计结果。"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from q1_model_fixed import EvaluationConfig, evaluate_field


BASE = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the repaired Q1 optical evaluator")
    parser.add_argument("--input", default=os.path.join(BASE, "附件.xlsx"))
    parser.add_argument("--output-dir", default=os.path.join(BASE, "output", "q1_final"))
    parser.add_argument("--shadow-grid", type=int, default=11)
    parser.add_argument("--truncation-grid", type=int, default=11)
    parser.add_argument("--sun-rays", type=int, default=128)
    parser.add_argument("--neighbor-radius", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    data = pd.read_excel(args.input)
    xs = data.iloc[:, 0].astype(float).to_numpy()
    ys = data.iloc[:, 1].astype(float).to_numpy()
    config = EvaluationConfig(
        shadow_grid=args.shadow_grid,
        truncation_grid=args.truncation_grid,
        sun_rays=args.sun_rays,
        neighbor_radius=args.neighbor_radius,
    )
    if config.shadow_grid != config.truncation_grid:
        raise ValueError("--shadow-grid and --truncation-grid must be equal in the repaired evaluator")

    print(
        "[修正版配置] "
        f"阴影网格={config.shadow_grid}x{config.shadow_grid}, "
        f"截断镜面网格={config.truncation_grid}x{config.truncation_grid}, "
        f"太阳盘光线={config.sun_rays}, 邻域半径={config.neighbor_radius:.0f} m"
    )
    print(f"[加载] 定日镜数量 N={len(xs)}，接收器中心 z={config.receiver_center_z:.1f} m")
    result = evaluate_field(xs, ys, config)
    elapsed = time.time() - started

    os.makedirs(args.output_dir, exist_ok=True)
    monthly = pd.DataFrame(result["monthly"])
    monthly_path = os.path.join(args.output_dir, "q1_table1_fixed.csv")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    summary = dict(result["summary"])
    summary["elapsed_seconds"] = elapsed
    summary_path = os.path.join(args.output_dir, "q1_summary_fixed.json")
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    time_records_path = os.path.join(args.output_dir, "q1_time_records_fixed.csv")
    pd.DataFrame(result["time_records"]).to_csv(
        time_records_path,
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "日期",
        "平均光学效率",
        "平均余弦效率",
        "平均阴影遮挡效率",
        "平均截断效率",
        "单位面积镜面平均输出热功率(kW/m2)",
    ]
    print("\n================ 问题一修正版结果 ================")
    print(monthly[display_columns].round(4).to_string(index=False))
    print("----------------------------------------------------")
    print(f"年平均光学效率              : {summary['annual_optical_efficiency']:.6f}")
    print(f"年平均余弦效率              : {summary['annual_cosine_efficiency']:.6f}")
    print(f"年平均阴影遮挡效率          : {summary['annual_shadow_blocking_efficiency']:.6f}")
    print(f"年平均截断效率              : {summary['annual_truncation_efficiency']:.6f}")
    print(f"年平均输出热功率            : {summary['annual_output_power_mw']:.6f} MW")
    print(f"单位镜面面积年平均输出热功率: {summary['annual_unit_area_power_kw_m2']:.6f} kW/m2")
    print(f"邻域候选镜数                : 平均 {summary['neighbor_count_mean']:.1f}，最大 {summary['neighbor_count_max']}")
    print(f"耗时                        : {elapsed:.1f} s")
    print(f"[输出] {monthly_path}")
    print(f"[输出] {summary_path}")
    print(f"[输出] {time_records_path}")


if __name__ == "__main__":
    main()
