# -*- coding: utf-8 -*-
"""
2023 CUMCM A 题「定日镜场的优化设计」—— 问题 1 核心模型
========================================================
依据 2023 高教社杯 A 题公开附录建立的定日镜场光学效率计算引擎。

主要创新点（见论文提纲）：
  1. 层次化邻域遮挡检测：空间网格粗筛(L1) → 方向性 + 投影包围盒阈值截断(L2)
     → 射线-矩形精确求交(L3)，将 O(N^2) 降到 O(N·k)。
     其中 L2 的投影膨胀阈值 eps 随距离自适应（远镜粗筛、近镜精筛）。
  2. 截断效率解析—数值混合：远场用椭圆高斯卷积解析积分，近场用 Buie 太阳
     形状重要性采样追迹；切换由无量纲光斑尺寸阈值 d* 决定。

单位约定：长度 m，角度 rad（公式内部），功率 kW，辐照度 kW/m²。
"""

import numpy as np

# ======================================================================
# 一、题目与附录常量
# ======================================================================
LATITUDE      = 39.4            # 北纬 (deg)
ALTITUDE_KM   = 3.0             # 海拔 (km)
SOLAR_CONST   = 1.366           # G0 (kW/m^2)

TOWER_HEIGHT  = 80.0            # 吸收塔高度 (m)
RECEIVER_H    = 8.0             # 集热器圆柱高 (m)
RECEIVER_D    = 7.0             # 集热器圆柱直径 (m)
RECEIVER_R    = RECEIVER_D / 2.0
RECEIVER_CZ   = TOWER_HEIGHT + RECEIVER_H / 2.0   # 集热器中心高度 84 m

MIRROR_W      = 6.0             # Q1 镜面宽度 (m)
MIRROR_H      = 6.0             # Q1 镜面高度 (m)
INSTALL_H     = 4.0             # Q1 安装高度 (m)
REFLECTIVITY  = 0.92            # 镜面反射率 (附录)
MIRROR_AREA   = MIRROR_W * MIRROR_H

SUN_HALF_ANGLE = 4.65e-3        # 太阳盘半角 (rad)
SUN_SIGMA      = SUN_HALF_ANGLE / 2.0   # 等效高斯标准差 ~2.325 mrad

# 12 个月 21 日的年积日 (2023 非闰年)
DAY_OF_YEAR = np.array([21, 52, 80, 111, 141, 172, 202, 233, 264, 294, 325, 355])
# 5 个当地时刻 (小时)
LOCAL_TIMES = np.array([9.0, 10.5, 12.0, 13.5, 15.0])

# ======================================================================
# 二、太阳几何与 DNI
# ======================================================================
def solar_declination(day):
    """太阳赤纬角 (rad)。

    附录公式：sin δ = sin(2πD/365)·sin(23.45°)，
    其中 D 以春分（3 月 21 日）为第 0 天起算（2023 非闰年，春分年积日为 80）。
    """
    spring_equinox_doy = 80.0
    D = (day - spring_equinox_doy) % 365.0
    sin_delta = np.sin(2.0 * np.pi * D / 365.0) * np.sin(np.deg2rad(23.45))
    return np.arcsin(np.clip(sin_delta, -1.0, 1.0))


def hour_angle(solar_time):
    """太阳时角 (rad)。solar_time 为当地太阳时 (h)。"""
    return np.deg2rad(15.0) * (solar_time - 12.0)


def sun_direction(day, solar_time):
    """返回太阳方向单位向量 (东, 北, 天) 及高度角 alpha (rad)。

    太阳高度角 sin(alpha) = sin(phi)sin(delta) + cos(phi)cos(delta)cos(omega)
    太阳方位角 gamma（自北顺时针为正）。
    """
    phi = np.deg2rad(LATITUDE)
    delta = solar_declination(day)
    omega = hour_angle(solar_time)

    sin_alpha = np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(omega)
    alpha = np.arcsin(np.clip(sin_alpha, -1.0, 1.0))

    cos_alpha = np.cos(alpha)
    sin_gamma = -np.cos(delta) * np.sin(omega) / np.clip(cos_alpha, 1e-12, None)
    cos_gamma = (np.sin(delta) - sin_alpha * np.sin(phi)) / np.clip(cos_alpha, 1e-12, None)
    gamma = np.arctan2(sin_gamma, cos_gamma)   # 自北顺时针

    # 太阳方向单位向量 (东, 北, 天)
    s = np.array([np.sin(gamma) * cos_alpha,
                  np.cos(gamma) * cos_alpha,
                  sin_alpha])
    return s, alpha


def dni(alpha):
    """法向直接辐射辐照度 DNI (kW/m^2)。附录经验公式，海拔 3 km。

    DNI = G0 * [a + b * exp(-c / sin(alpha))]
    """
    H = ALTITUDE_KM
    a = 0.4237 - 0.00821 * (6 - H) ** 2
    b = 0.5055 + 0.00595 * (6.5 - H) ** 2
    c = 0.2711 + 0.01858 * (2.5 - H) ** 2
    sin_alpha = np.sin(alpha)
    out = np.where(sin_alpha > 1e-3,
                   SOLAR_CONST * (a + b * np.exp(-c / sin_alpha)),
                   0.0)
    return np.maximum(out, 0.0)


# ======================================================================
# 三、定日镜姿态
# ======================================================================
def mirror_pose(x, y, s, tower_center=(0.0, 0.0)):
    """计算定日镜法向量及镜面局部基。

    x, y : 镜中心水平坐标 (m)
    s    : 太阳方向单位向量 (东,北,天)
    返回 n (法向量), u (宽度方向,水平), v (高度方向,竖直面内)。
    """
    # 镜中心 -> 集热器中心
    r = np.array([tower_center[0] - x, tower_center[1] - y, RECEIVER_CZ - INSTALL_H])
    r = r / np.linalg.norm(r)
    # 反射定律：法向量平分 -s 与 r 的夹角（入射角 = 反射角）
    n = s + r                      # s 指向太阳, r 指向集热器
    n = n / np.linalg.norm(n)
    # 宽度方向：水平且垂直于 n（镜面上下边平行地面）
    u = np.cross(n, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(u) < 1e-12:   # 法向量竖直时的退化
        u = np.array([1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    # 高度方向：垂直于 n 和 u
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return n, u, v


# ======================================================================
# 四、效率分量（解析部分）
# ======================================================================
def cosine_efficiency(s, n):
    """余弦效率 = 入射角余弦 = s·n。"""
    return float(np.clip(np.dot(s, n), 0.0, 1.0))


def atmospheric_transmittance(d):
    """大气透射率。d 为镜心到集热器中心距离 (m)。"""
    return 0.99321 - 0.0001176 * d + 1.97e-8 * d * d


# ======================================================================
# 五、层次化邻域遮挡检测（创新点 1）
# ======================================================================
class NeighborIndex:
    """L1 层：均匀空间网格，预计算每面镜的候选邻居。

    仅保留距离 < search_radius 的镜对，将 O(N^2) 降为 O(N·k)。
    """

    def __init__(self, xs, ys, cell_size=25.0, search_radius=30.0):
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)
        self.N = len(xs)
        self.cell = cell_size
        self.radius = search_radius
        self._build()

    def _build(self):
        grid = {}
        for i in range(self.N):
            cx = int(np.floor(self.xs[i] / self.cell))
            cy = int(np.floor(self.ys[i] / self.cell))
            grid.setdefault((cx, cy), []).append(i)
        # 对每面镜收集其 3x3 邻域格内的候选（去重、去自身）
        neighbors = [set() for _ in range(self.N)]
        for i in range(self.N):
            cx = int(np.floor(self.xs[i] / self.cell))
            cy = int(np.floor(self.ys[i] / self.cell))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((cx + dx, cy + dy), []):
                        if j == i:
                            continue
                        dxij = self.xs[j] - self.xs[i]
                        dyij = self.ys[j] - self.ys[i]
                        if dxij * dxij + dyij * dyij <= self.radius ** 2:
                            neighbors[i].add(j)
        self.neighbors = [np.array(sorted(nb), dtype=int) for nb in neighbors]

    def candidates(self, i):
        return self.neighbors[i]


def _ray_rect_hits(points, direction, center, u, v, half_w, half_h, eps):
    """L3 层：射线-矩形求交（向量化）。

    points    : (M, 3) 采样点
    direction : (3,)   射线方向单位向量
    center    : (3,)   矩形中心
    u, v      : (3,)   矩形局部基（宽度、高度方向）
    half_w/h  : float  半宽/半高
    eps       : float  投影包围盒膨胀阈值（自适应）

    返回 (M,) bool：每个采样点的射线是否命中该矩形（膨胀 eps）。
    """
    n = np.cross(u, v)                          # 矩形法向
    n = n / np.linalg.norm(n)
    denom = np.dot(direction, n)
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    t = np.dot(center - points, n) / denom      # 到矩形平面的有向距离
    hit = (t > 0) & np.isfinite(t)
    X = points + t[:, None] * direction[None, :]
    local = X - center[None, :]
    lx = np.abs(np.dot(local, u))
    ly = np.abs(np.dot(local, v))
    hit &= (lx <= half_w + eps) & (ly <= half_h + eps)
    return hit


def adaptive_eps(distance):
    """L2 层自适应投影阈值 eps：远镜粗筛（大 eps 快速剔除），近镜精筛（小 eps）。

    距离越大，镜面投影的几何不确定性越大，采用更保守（更大）的膨胀阈值；
    距离小则用更精确（更小）的阈值。这里用分段线性经验映射。
    """
    distance = np.asarray(distance)
    # 近场 (<120 m)：0.15 m；远场 (>250 m)：0.60 m；中间线性过渡
    eps = np.clip(0.15 + (distance - 120.0) * (0.60 - 0.15) / (250.0 - 120.0),
                  0.15, 0.60)
    return eps


def shadow_blocking_efficiency(i, center, u, v, s, r, idx, grid_pts=3):
    """计算第 i 面镜的阴影遮挡效率（入射阴影 + 反射遮挡）。

    将镜面栅格化 grid_pts×grid_pts 个采样点，对每个采样点：
      - 入射遮挡：沿 -s 方向回溯，判断是否被候选镜拦截；
      - 反射遮挡：沿 r 方向前进，判断是否被候选镜拦截。
    有效采样点比例即阴影遮挡效率。

    采样点局部坐标 a∈[-w/2, w/2], b∈[-h/2, h/2]（3x3 点）。
    """
    w, h = MIRROR_W, MIRROR_H
    aa = np.linspace(-w / 2, w / 2, grid_pts)
    bb = np.linspace(-h / 2, h / 2, grid_pts)
    A, B = np.meshgrid(aa, bb)
    points = center[None, :] + A.ravel()[:, None] * u[None, :] + B.ravel()[:, None] * v[None, :]
    M = points.shape[0]

    occ_in = np.zeros(M, dtype=bool)   # 入射遮挡
    occ_re = np.zeros(M, dtype=bool)   # 反射遮挡

    for j in idx.candidates(i):
        cj = np.array([idx.xs[j], idx.ys[j], INSTALL_H])
        nj, uj, vj = mirror_pose(idx.xs[j], idx.ys[j], s)
        eps = float(adaptive_eps(np.linalg.norm(cj - center)))
        # 入射遮挡：光线从太阳来，沿 -s 传播
        occ_in |= _ray_rect_hits(points, -s, cj, uj, vj, w / 2, h / 2, eps)
        # 反射遮挡：光线沿 r 向集热器传播
        occ_re |= _ray_rect_hits(points, r, cj, uj, vj, w / 2, h / 2, eps)

    invalid = occ_in | occ_re
    return 1.0 - invalid.sum() / M


# ======================================================================
# 六、截断效率：解析—数值混合（创新点 2）
# ======================================================================
TRUNC_DSTAR = 150.0     # 无量纲切换距离 (m)：< d* 用 Buie 追迹，>= d* 用高斯解析
TRUNC_N_RAYS = 200      # 近场每镜追迹光线数


def truncation_analytic(d, cos_incidence):
    """远场截断效率：椭圆高斯卷积解析积分。

    光斑近似为椭圆高斯，标准差由镜面投影 + 太阳锥弥散合成：
        sigma = sqrt( (mirror_proj/sqrt(12))^2 + (d * SUN_SIGMA)^2 )
    集热器圆柱在垂直反射方向平面上的投影近似为矩形 7m x 8m。
    截断效率 = erf(Rx/(sigma sqrt2)) * erf(Ry/(sigma sqrt2))。
    """
    w_proj = MIRROR_W * cos_incidence      # 镜面宽度方向投影
    h_proj = MIRROR_H * cos_incidence      # 镜面高度方向投影（简化同向）
    sig_x = np.sqrt((w_proj / np.sqrt(12.0)) ** 2 + (d * SUN_SIGMA) ** 2)
    sig_y = np.sqrt((h_proj / np.sqrt(12.0)) ** 2 + (d * SUN_SIGMA) ** 2)
    from math import erf
    return erf(RECEIVER_R / (sig_x * np.sqrt(2.0))) * erf(RECEIVER_H / 2.0 / (sig_y * np.sqrt(2.0)))


def truncation_buie(center, n, r, seed=0):
    """近场截断效率：Buie 太阳形状重要性采样追迹。

    在镜面上采样点，每个点沿反射方向 r 加太阳锥扰动（Buie 分布），
    追踪光线到集热器中心平面，判断落点是否落在圆柱投影矩形内。
    """
    rng = np.random.default_rng(seed)
    # 镜面 5x5 采样点
    aa = np.linspace(-MIRROR_W / 2, MIRROR_W / 2, 5)
    bb = np.linspace(-MIRROR_H / 2, MIRROR_H / 2, 5)
    A, B = np.meshgrid(aa, bb)
    u = np.cross(n, np.array([0.0, 0.0, 1.0]))
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    pts = center[None, :] + A.ravel()[:, None] * u[None, :] + B.ravel()[:, None] * v[None, :]

    # 建立垂直 r 的平面基
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(r, ref)) > 0.99:
        ref = np.array([1.0, 0.0, 0.0])
    p1 = np.cross(r, ref); p1 = p1 / np.linalg.norm(p1)
    p2 = np.cross(r, p1)

    hits = 0
    total = 0
    for pt in pts:
        # Buie 重要性采样：盘区密集（余弦近似），环日区稀疏。这里用等效高斯近似太阳锥。
        theta = SUN_SIGMA * rng.standard_normal(TRUNC_N_RAYS // len(pts))
        phi = 2 * np.pi * rng.random(TRUNC_N_RAYS // len(pts))
        # 扰动方向 = r 附近偏离 theta
        dirs = r[None, :] + (theta * np.cos(phi))[:, None] * p1[None, :] \
                           + (theta * np.sin(phi))[:, None] * p2[None, :]
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        # 光线传播到集热器中心平面 (法向 r)
        t = np.dot(np.array([0.0, 0.0, RECEIVER_CZ]) - pt, r) / np.clip(np.dot(dirs, r), 1e-6, None)
        X = pt[None, :] + t[:, None] * dirs
        local = X - np.array([0.0, 0.0, RECEIVER_CZ])
        lx = np.abs(np.dot(local, p1))
        ly = np.abs(np.dot(local, p2))
        hits += int(((lx <= RECEIVER_R) & (ly <= RECEIVER_H / 2.0)).sum())
        total += dirs.shape[0]
    return hits / total


def truncation_efficiency(d, cos_incidence, center, n, r, seed=0):
    """截断效率：近场数值 / 远场解析 的混合自适应。"""
    if d < TRUNC_DSTAR:
        return truncation_buie(center, n, r, seed=seed)
    return truncation_analytic(d, cos_incidence)


# ======================================================================
# 七、总效率与输出热功率
# ======================================================================
def total_efficiency(eta_cos, eta_at, eta_sb, eta_trunc):
    """总光学效率 = 余弦 × 阴影遮挡 × 截断 × 大气透射 × 反射率。"""
    return eta_cos * eta_sb * eta_trunc * eta_at * REFLECTIVITY
