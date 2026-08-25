# -*- coding: utf-8 -*-
"""汇总问题一原模型与修正版的年度和月度差异。"""

from __future__ import annotations

import json
import os

import pandas as pd


BASE = os.path.dirname(os.path.abspath(__file__))
FIXED_DIR = os.path.join(BASE, "output", "q1_final")


def main() -> None:
    with open(os.path.join(BASE, "q1_summary_original.json"), encoding="utf-8") as stream:
        original = json.load(stream)
    with open(os.path.join(FIXED_DIR, "q1_summary_fixed.json"), encoding="utf-8") as stream:
        fixed = json.load(stream)

    metrics = [
        ("年平均光学效率", "annual_optical_efficiency"),
        ("年平均余弦效率", "annual_cosine_efficiency"),
        ("年平均阴影遮挡效率", "annual_shadow_blocking_efficiency"),
        ("年平均截断效率", "annual_truncation_efficiency"),
        ("年平均输出热功率(MW)", "annual_output_power_mw"),
        ("单位面积年平均输出热功率(kW/m2)", "annual_unit_area_power_kw_m2"),
    ]
    rows = []
    for label, key in metrics:
        before = float(original[key])
        after = float(fixed[key])
        rows.append({
            "指标": label,
            "改前": before,
            "改后": after,
            "绝对变化": after - before,
            "相对变化(%)": (after / before - 1.0) * 100.0 if before else None,
        })
    annual = pd.DataFrame(rows)
    annual_path = os.path.join(FIXED_DIR, "q1_before_after_annual.csv")
    annual.to_csv(annual_path, index=False, encoding="utf-8-sig")

    original_monthly = pd.read_csv(os.path.join(BASE, "q1_table1.csv"))
    fixed_monthly = pd.read_csv(os.path.join(FIXED_DIR, "q1_table1_fixed.csv"))
    original_monthly["日期"] = original_monthly["日期"].astype(str).str.replace("月$", "月21日", regex=True)
    monthly = original_monthly.merge(fixed_monthly, on="日期", suffixes=("_改前", "_改后"))
    monthly_path = os.path.join(FIXED_DIR, "q1_before_after_monthly.csv")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    convergence_inputs = [
        ("7x7/32 rays, radius 45 m", os.path.join(BASE, "output", "q1_sensitivity", "r45_g7_t7_s32", "q1_summary_fixed.json")),
        ("7x7/32 rays, radius 60 m", os.path.join(BASE, "output", "q1_fixed", "q1_summary_fixed.json")),
        ("7x7/32 rays, radius 90 m", os.path.join(BASE, "output", "q1_sensitivity", "r90_g7_t7_s32", "q1_summary_fixed.json")),
        ("9x9/64 rays, radius 60 m", os.path.join(BASE, "output", "q1_sensitivity", "r60_g9_t9_s64", "q1_summary_fixed.json")),
        ("11x11/128 rays, radius 60 m", os.path.join(BASE, "output", "q1_final", "q1_summary_fixed.json")),
    ]
    convergence_rows = []
    for label, path in convergence_inputs:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as stream:
            item = json.load(stream)
        convergence_rows.append({
            "配置": label,
            "年平均光学效率": item["annual_optical_efficiency"],
            "年平均阴影遮挡效率": item["annual_shadow_blocking_efficiency"],
            "年平均截断效率": item["annual_truncation_efficiency"],
            "年平均输出热功率(MW)": item["annual_output_power_mw"],
            "单位面积功率(kW/m2)": item["annual_unit_area_power_kw_m2"],
        })
    convergence_path = os.path.join(FIXED_DIR, "q1_convergence.csv")
    pd.DataFrame(convergence_rows).to_csv(convergence_path, index=False, encoding="utf-8-sig")

    print("\n================ 问题一改前 / 改后 ================")
    print(annual.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\n[输出] {annual_path}")
    print(f"[输出] {monthly_path}")
    print(f"[输出] {convergence_path}")


if __name__ == "__main__":
    main()
