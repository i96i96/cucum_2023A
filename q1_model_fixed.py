# -*- coding: utf-8 -*-
"""问题一修正版光学评价器。

相对原模型的关键修正：
1. 题面把吸收塔高度定义为集热器中心离地高度，因此接收器中心 z=80 m；
2. 阴影检测从目标镜面朝太阳方向（+s）回溯，而不是沿光传播方向（-s）继续前进；
3. 阴影/遮挡采用无膨胀的确定性中点网格，并使用可配置的安全邻域半径；
4. 截断效率统一使用确定性低差异太阳盘采样和真实竖直圆柱侧面求交，取消 150 m 分段。

单位：长度 m，角度 rad，功率 kW，辐照度 kW/m^2。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from q1_model import (
    DAY_OF_YEAR,
    LOCAL_TIMES,
    REFLECTIVITY,
    dni,
    sun_direction,
)


TOWER_HEIGHT = 80.0
RECEIVER_CENTER_Z = 80.0
RECEIVER_HEIGHT = 8.0
RECEIVER_RADIUS = 3.5
SUN_HALF_ANGLE = 4.65e-3

MIRROR_WIDTH = 6.0
MIRROR_HEIGHT = 6.0
INSTALL_HEIGHT = 4.0
MIRROR_AREA = MIRROR_WIDTH * MIRROR_HEIGHT


@dataclass(frozen=True)
class EvaluationConfig:
    """修正版评价器的精度和几何配置。"""

    tower_x: float = 0.0
    tower_y: float = 0.0
    receiver_center_z: float = RECEIVER_CENTER_Z
    receiver_height: float = RECEIVER_HEIGHT
    receiver_radius: float = RECEIVER_RADIUS
    mirror_width: float = MIRROR_WIDTH
    mirror_height: float = MIRROR_HEIGHT
    install_height: float = INSTALL_HEIGHT
    reflectivity: float = REFLECTIVITY
    shadow_grid: int = 11
    truncation_grid: int = 11
    sun_rays: int = 128
    sun_half_angle: float = SUN_HALF_ANGLE
    neighbor_radius: float = 60.0
    neighbor_cell_size: float = 25.0
    truncation_batch_size: int = 64

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)


class NeighborIndex:
    """均匀网格邻域索引，支持任意搜索半径。"""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, radius: float, cell_size: float = 25.0):
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)
        self.radius = float(radius)
        self.cell_size = float(cell_size)
        self._neighbors = self._build()

    def _build(self) -> Tuple[np.ndarray, ...]:
        grid: Dict[Tuple[int, int], list[int]] = {}
        cells = np.floor(np.column_stack([self.xs, self.ys]) / self.cell_size).astype(int)
        for i, (cx, cy) in enumerate(cells):
            grid.setdefault((int(cx), int(cy)), []).append(i)

        span = int(np.ceil(self.radius / self.cell_size))
        radius_sq = self.radius * self.radius
        neighbors = []
        for i, (cx, cy) in enumerate(cells):
            found = []
            for gx in range(int(cx) - span, int(cx) + span + 1):
                for gy in range(int(cy) - span, int(cy) + span + 1):
                    for j in grid.get((gx, gy), ()):  # local candidates only
                        if j == i:
                            continue
                        dx = self.xs[j] - self.xs[i]
                        dy = self.ys[j] - self.ys[i]
                        if dx * dx + dy * dy <= radius_sq:
                            found.append(j)
            neighbors.append(np.asarray(sorted(set(found)), dtype=int))
        return tuple(neighbors)

    def candidates(self, i: int) -> np.ndarray:
        return self._neighbors[i]

    @property
    def counts(self) -> np.ndarray:
        return np.asarray([len(items) for items in self._neighbors], dtype=int)


def atmospheric_transmittance(distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    return 0.99321 - 0.0001176 * distance + 1.97e-8 * distance * distance


def _unit(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / np.maximum(norms, 1e-15)


def mirror_poses(
    xs: np.ndarray,
    ys: np.ndarray,
    sun_vector: np.ndarray,
    config: EvaluationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回镜心、法向、镜面局部基、中心反射方向和镜心到接收器距离。"""

    n_mirrors = len(xs)
    centers = np.column_stack([
        xs,
        ys,
        np.full(n_mirrors, config.install_height, dtype=float),
    ])
    receiver = np.array([
        config.tower_x,
        config.tower_y,
        config.receiver_center_z,
    ])
    to_receiver = receiver[None, :] - centers
    distances = np.linalg.norm(to_receiver, axis=1)
    reflected = to_receiver / np.maximum(distances, 1e-15)[:, None]

    normals = _unit(sun_vector[None, :] + reflected)
    vertical = np.array([0.0, 0.0, 1.0])
    basis_u = np.cross(normals, vertical[None, :])
    degenerate = np.linalg.norm(basis_u, axis=1) < 1e-12
    basis_u[degenerate] = np.array([1.0, 0.0, 0.0])
    basis_u = _unit(basis_u)
    basis_v = _unit(np.cross(normals, basis_u))
    return centers, normals, basis_u, basis_v, reflected, distances


def midpoint_grid(width: float, height: float, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """矩形面内等面积中点网格；避开边界点造成的判定偏差。"""

    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    fractions = (np.arange(grid_size, dtype=float) + 0.5) / grid_size - 0.5
    aa, bb = np.meshgrid(width * fractions, height * fractions, indexing="xy")
    return aa.ravel(), bb.ravel()


def mirror_surface_points(
    centers: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    width: float,
    height: float,
    grid_size: int,
) -> np.ndarray:
    aa, bb = midpoint_grid(width, height, grid_size)
    return (
        centers[:, None, :]
        + aa[None, :, None] * basis_u[:, None, :]
        + bb[None, :, None] * basis_v[:, None, :]
    )


def _ray_rectangle_hits(
    points: np.ndarray,
    direction: np.ndarray,
    candidate_centers: np.ndarray,
    candidate_normals: np.ndarray,
    candidate_u: np.ndarray,
    candidate_v: np.ndarray,
    half_width: float,
    half_height: float,
) -> np.ndarray:
    """判断目标镜采样点的平行射线是否命中任一候选矩形。"""

    if len(candidate_centers) == 0:
        return np.zeros(len(points), dtype=bool)

    direction = np.asarray(direction, dtype=float)
    denominators = candidate_normals @ direction
    deltas = candidate_centers[:, None, :] - points[None, :, :]
    numerators = np.einsum("kmi,ki->km", deltas, candidate_normals)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = numerators / denominators[:, None]

    intersections = points[None, :, :] + t[:, :, None] * direction[None, None, :]
    local = intersections - candidate_centers[:, None, :]
    local_u = np.abs(np.einsum("kmi,ki->km", local, candidate_u))
    local_v = np.abs(np.einsum("kmi,ki->km", local, candidate_v))
    hits = (
        (t > 1e-8)
        & np.isfinite(t)
        & (np.abs(denominators)[:, None] > 1e-12)
        & (local_u <= half_width + 1e-9)
        & (local_v <= half_height + 1e-9)
    )
    return np.any(hits, axis=0)


def shadow_blocking_visibility(
    centers: np.ndarray,
    normals: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    sun_vector: np.ndarray,
    reflected: np.ndarray,
    index: NeighborIndex,
    config: EvaluationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回阴影遮挡效率及每个镜面采样点的可用掩码。"""

    points_all = mirror_surface_points(
        centers,
        basis_u,
        basis_v,
        config.mirror_width,
        config.mirror_height,
        config.shadow_grid,
    )
    efficiencies = np.empty(len(centers), dtype=float)
    visibility = np.empty((len(centers), points_all.shape[1]), dtype=bool)
    half_width = config.mirror_width / 2.0
    half_height = config.mirror_height / 2.0

    for i in range(len(centers)):
        candidates = index.candidates(i)
        points = points_all[i]
        # 从目标镜面朝太阳回溯才是在检查太阳与目标之间的遮挡物。
        shaded = _ray_rectangle_hits(
            points,
            sun_vector,
            centers[candidates],
            normals[candidates],
            basis_u[candidates],
            basis_v[candidates],
            half_width,
            half_height,
        )
        blocked = _ray_rectangle_hits(
            points,
            reflected[i],
            centers[candidates],
            normals[candidates],
            basis_u[candidates],
            basis_v[candidates],
            half_width,
            half_height,
        )
        visibility[i] = ~(shaded | blocked)
        efficiencies[i] = np.mean(visibility[i])
    return efficiencies, visibility


def _van_der_corput(indices: Iterable[int], base: int) -> np.ndarray:
    values = []
    for index in indices:
        n = int(index)
        denominator = 1.0
        value = 0.0
        while n:
            n, remainder = divmod(n, base)
            denominator *= base
            value += remainder / denominator
        values.append(value)
    return np.asarray(values, dtype=float)


def solar_disc_directions(
    sun_vector: np.ndarray,
    count: int,
    half_angle: float,
) -> np.ndarray:
    """用 Halton(2,3) 低差异点均匀采样题面给定的太阳锥截面。"""

    if count < 1:
        raise ValueError("count must be positive")
    indices = np.arange(1, count + 1)
    radial_u = _van_der_corput(indices, 2)
    azimuth_u = _van_der_corput(indices, 3)
    theta = half_angle * np.sqrt(radial_u)
    azimuth = 2.0 * np.pi * azimuth_u

    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(sun_vector, reference)) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    axis_1 = _unit(np.cross(sun_vector, reference))
    axis_2 = _unit(np.cross(sun_vector, axis_1))
    offsets = (
        np.tan(theta)[:, None] * np.cos(azimuth)[:, None] * axis_1[None, :]
        + np.tan(theta)[:, None] * np.sin(azimuth)[:, None] * axis_2[None, :]
    )
    return _unit(sun_vector[None, :] + offsets)


def ray_hits_receiver_cylinder(
    origins: np.ndarray,
    directions: np.ndarray,
    config: EvaluationConfig,
) -> np.ndarray:
    """求射线与竖直圆柱侧面的首次正向交点，并检查交点高度。"""

    rel_x = origins[:, :, 0] - config.tower_x
    rel_y = origins[:, :, 1] - config.tower_y
    dir_x = directions[:, :, 0]
    dir_y = directions[:, :, 1]

    a = dir_x * dir_x + dir_y * dir_y
    b = 2.0 * (
        rel_x[:, :, None] * dir_x[:, None, :]
        + rel_y[:, :, None] * dir_y[:, None, :]
    )
    c = rel_x * rel_x + rel_y * rel_y - config.receiver_radius ** 2
    discriminant = b * b - 4.0 * a[:, None, :] * c[:, :, None]

    valid_disc = discriminant >= 0.0
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
    denominator = 2.0 * np.maximum(a[:, None, :], 1e-18)
    t_near = (-b - sqrt_disc) / denominator
    z_near = origins[:, :, 2, None] + t_near * directions[:, None, :, 2]
    lower = config.receiver_center_z - config.receiver_height / 2.0
    upper = config.receiver_center_z + config.receiver_height / 2.0
    hit_x = rel_x[:, :, None] + t_near * dir_x[:, None, :]
    hit_y = rel_y[:, :, None] + t_near * dir_y[:, None, :]
    # 外表受光：光线方向必须指向圆柱外法线的反方向；远根对应从内向外撞击背面。
    front_facing = hit_x * dir_x[:, None, :] + hit_y * dir_y[:, None, :] < 0.0
    return (
        valid_disc
        & (t_near > 1e-8)
        & (z_near >= lower)
        & (z_near <= upper)
        & front_facing
    )


def truncation_efficiencies(
    centers: np.ndarray,
    normals: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    sun_vector: np.ndarray,
    visibility: np.ndarray,
    config: EvaluationConfig,
) -> np.ndarray:
    """在未被阴影/遮挡的镜面区域上条件计算截断效率。"""

    sun_samples = solar_disc_directions(
        sun_vector,
        config.sun_rays,
        config.sun_half_angle,
    )
    incoming = -sun_samples
    efficiencies = np.empty(len(centers), dtype=float)

    for start in range(0, len(centers), config.truncation_batch_size):
        stop = min(start + config.truncation_batch_size, len(centers))
        batch_normals = normals[start:stop]
        dot_in_normal = np.einsum("kj,bj->bk", incoming, batch_normals)
        outgoing = (
            incoming[None, :, :]
            - 2.0 * dot_in_normal[:, :, None] * batch_normals[:, None, :]
        )
        outgoing = _unit(outgoing)
        origins = mirror_surface_points(
            centers[start:stop],
            basis_u[start:stop],
            basis_v[start:stop],
            config.mirror_width,
            config.mirror_height,
            config.truncation_grid,
        )
        hits = ray_hits_receiver_cylinder(origins, outgoing, config)
        batch_visibility = visibility[start:stop]
        if batch_visibility.shape[1] != hits.shape[1]:
            raise ValueError("shadow_grid and truncation_grid must be equal")
        usable_hits = hits & batch_visibility[:, :, None]
        denominators = np.sum(batch_visibility, axis=1) * hits.shape[2]
        efficiencies[start:stop] = np.divide(
            np.sum(usable_hits, axis=(1, 2)),
            denominators,
            out=np.zeros(stop - start, dtype=float),
            where=denominators > 0,
        )
    return efficiencies


def evaluate_one_time(
    xs: np.ndarray,
    ys: np.ndarray,
    sun_vector: np.ndarray,
    index: NeighborIndex,
    config: EvaluationConfig,
) -> Dict[str, np.ndarray]:
    centers, normals, basis_u, basis_v, reflected, distances = mirror_poses(
        xs, ys, sun_vector, config
    )
    cosine = np.clip(normals @ sun_vector, 0.0, 1.0)
    atmosphere = atmospheric_transmittance(distances)
    shadow_blocking, visibility = shadow_blocking_visibility(
        centers,
        normals,
        basis_u,
        basis_v,
        sun_vector,
        reflected,
        index,
        config,
    )
    truncation = truncation_efficiencies(
        centers,
        normals,
        basis_u,
        basis_v,
        sun_vector,
        visibility,
        config,
    )
    optical = (
        cosine
        * atmosphere
        * shadow_blocking
        * truncation
        * config.reflectivity
    )
    return {
        "optical": optical,
        "cosine": cosine,
        "shadow_blocking": shadow_blocking,
        "truncation": truncation,
        "atmosphere": atmosphere,
    }


def evaluate_field(
    xs: np.ndarray,
    ys: np.ndarray,
    config: EvaluationConfig | None = None,
) -> Dict[str, object]:
    """计算题目规定的 12 月 x 5 时刻并返回月度及年平均指标。"""

    config = config or EvaluationConfig()
    if config.shadow_grid != config.truncation_grid:
        raise ValueError("shadow_grid and truncation_grid must be equal")
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) != len(ys) or len(xs) == 0:
        raise ValueError("xs and ys must be non-empty arrays of equal length")

    index = NeighborIndex(
        xs,
        ys,
        radius=config.neighbor_radius,
        cell_size=config.neighbor_cell_size,
    )
    n_mirrors = len(xs)
    mirror_area = config.mirror_width * config.mirror_height
    monthly = []
    all_time_records = []

    month_names = [f"{month}月21日" for month in range(1, 13)]
    for month_index, day in enumerate(DAY_OF_YEAR):
        month_records = []
        for solar_time in LOCAL_TIMES:
            sun_vector, alpha = sun_direction(day, solar_time)
            direct_normal_irradiance = float(dni(alpha))
            components = evaluate_one_time(xs, ys, sun_vector, index, config)
            total_power_kw = direct_normal_irradiance * mirror_area * np.sum(components["optical"])
            record = {
                "month_index": month_index,
                "solar_time": float(solar_time),
                "dni": direct_normal_irradiance,
                "optical": float(np.mean(components["optical"])),
                "cosine": float(np.mean(components["cosine"])),
                "shadow_blocking": float(np.mean(components["shadow_blocking"])),
                "truncation": float(np.mean(components["truncation"])),
                "atmosphere": float(np.mean(components["atmosphere"])),
                "total_power_kw": float(total_power_kw),
                "unit_area_power_kw_m2": float(total_power_kw / (n_mirrors * mirror_area)),
            }
            month_records.append(record)
            all_time_records.append(record)

        monthly.append({
            "日期": month_names[month_index],
            "平均光学效率": float(np.mean([r["optical"] for r in month_records])),
            "平均余弦效率": float(np.mean([r["cosine"] for r in month_records])),
            "平均阴影遮挡效率": float(np.mean([r["shadow_blocking"] for r in month_records])),
            "平均截断效率": float(np.mean([r["truncation"] for r in month_records])),
            "单位面积镜面平均输出热功率(kW/m2)": float(
                np.mean([r["unit_area_power_kw_m2"] for r in month_records])
            ),
            "平均输出热功率(MW)": float(
                np.mean([r["total_power_kw"] for r in month_records]) / 1000.0
            ),
        })

    summary = {
        "model": "fixed",
        "receiver_center_z_m": float(config.receiver_center_z),
        "mirror_count": int(n_mirrors),
        "mirror_area_total_m2": float(n_mirrors * mirror_area),
        "annual_optical_efficiency": float(np.mean([r["optical"] for r in all_time_records])),
        "annual_cosine_efficiency": float(np.mean([r["cosine"] for r in all_time_records])),
        "annual_shadow_blocking_efficiency": float(
            np.mean([r["shadow_blocking"] for r in all_time_records])
        ),
        "annual_truncation_efficiency": float(
            np.mean([r["truncation"] for r in all_time_records])
        ),
        "annual_atmospheric_efficiency": float(
            np.mean([r["atmosphere"] for r in all_time_records])
        ),
        "annual_output_power_mw": float(
            np.mean([r["total_power_kw"] for r in all_time_records]) / 1000.0
        ),
        "annual_unit_area_power_kw_m2": float(
            np.mean([r["unit_area_power_kw_m2"] for r in all_time_records])
        ),
        "neighbor_count_mean": float(np.mean(index.counts)),
        "neighbor_count_max": int(np.max(index.counts)),
        "config": config.to_dict(),
    }
    return {
        "monthly": monthly,
        "summary": summary,
        "time_records": all_time_records,
    }
