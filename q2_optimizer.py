# -*- coding: utf-8 -*-
"""
2023 CUMCM A 题问题二 —— 多保真度信任域 + 保真度切换算法（封装）
================================================================
基于预检验结论设计的「变参数化层次信任域」优化器：

  预检验结论（q2_precheck.csv）：
    * L0 与细层级相关性 0.88–0.91，可用于全局探索，但精细排序不可靠；
    * L1/L2/L3 高度一致（Spearman ≥ 0.965），细层级族内切换近乎无损；
    * 保真度跳跃最大的位置在 L0→L1（径向分段引入主导自由度）。

  据此，算法策略：
    * 从 L0 起步做全局探索（平滑、低维、鲁棒）；
    * 粗层级信任域收敛（半径 < delta_switch）即「提升」到 L1，再提升到 L2；
    * 细层级停滞时「降级」回 L0 随机重启，以跳出局部最优；
    * 层级间是投影嵌套（f_fine(P(x)) = f_coarse(x)），提升点严格一致，
      无需额外修正项（符合 Alexandrov 模型管理 / 空间映射的一致性要求）。

单层级采用「线性模型信任域」：前向差分估计梯度 → 在无穷范数信任域内解线性
子问题（即 Cauchy 步）→ 实际/预测下降比 ρ 决定接受与半径缩放。

设计向量 x 在归一化空间 [0,1]^DIM 内操作；整数变量 K 的差分步长自动放大。
"""

import numpy as np
from dataclasses import dataclass, field
from q2_layout import (DIM, VAR_LO, VAR_HI, ACTIVE, INTEGER_VARS, N_GLOBAL,
                       to_unit, from_unit, MultiLevelLayout)


@dataclass
class TRConfig:
    """信任域 + 切换算法的全部超参数。"""
    # ---- 单层级信任域 ----
    eta_accept: float = 0.1        # 接受阈值：ρ ≥ η1 才接受试步
    eta_expand: float = 0.7        # 扩展阈值：ρ ≥ η2 扩大半径
    gamma_inc: float = 2.0         # 半径扩大因子
    gamma_dec: float = 0.5         # 半径收缩因子
    delta_init: float = 0.25       # 初始信任半径（归一化）
    delta_max: float = 1.0         # 半径上限
    delta_min: float = 0.02        # 局部收敛半径（最细层级以此判停）
    fd_step: float = 0.01          # 前向差分步长（归一化；整数变量自动放大）
    flat_grad_eps: float = 1e-6    # 梯度近零判定（平台期触发随机探索）
    # ---- 保真度切换 ----
    delta_switch: float = 0.05     # 粗层级半径低于此值 → 提升保真度
    stall_accept: int = 5          # 细层级连续停滞次数 → 降级逃逸
    # ---- 资源 ----
    max_iter: int = 400            # 总迭代上限
    max_evals: int = 5000          # 总评估预算（正式解题用 annual 模式时关键）
    lift_perturb: float = 0.0      # 提升时对新自由度的小扰动（0=严格投影）
    frozen_vars: tuple = ()          # 始终冻结的变量索引（如固定 tower_x=0）


class MultiFidelityTR:
    """变参数化层次信任域优化器（最大化目标）。"""

    def __init__(self, objective, config=None, rng=None):
        self.obj = objective                     # callable (x_physical, level) -> float
        self.cfg = config or TRConfig()
        self.rng = rng or np.random.default_rng(0)
        self.evals = 0
        self.history = []                        # 每步: (level, x, f, delta, rho, accepted)
        self.best = None                         # (u, f, level)
        self.layout = MultiLevelLayout()
        self.frozen = set(self.cfg.frozen_vars)

    # ------------------------------------------------------------------
    def _f(self, u, level):
        """归一化 u -> 物理 x -> 目标值。"""
        x = from_unit(u)
        val = float(self.obj(x, level))
        self.evals += 1
        return val

    def _gradient(self, u, f_u, level):
        """前向差分梯度（仅激活维；整数变量放大步长）。"""
        g = np.zeros(DIM)
        for i in ACTIVE[level]:
            if i in self.frozen:
                continue
            hi = self.cfg.fd_step
            if i in INTEGER_VARS:
                hi = max(hi, 1.0 / (VAR_HI[i] - VAR_LO[i]))   # K 至少变化 1
            uu = u.copy()
            uu[i] = min(u[i] + hi, 1.0)
            if uu[i] <= u[i] + 1e-12:                        # 撞上界，反向
                uu[i] = max(u[i] - hi, 0.0)
            denom = uu[i] - u[i]
            if abs(denom) < 1e-12:
                g[i] = 0.0
            else:
                g[i] = (self._f(uu, level) - f_u) / denom
        return g

    def _tr_subproblem(self, g, u, delta, level):
        """无穷范数信任域内最大化线性模型：Cauchy 步。"""
        s = np.zeros(DIM)
        pred = 0.0
        for i in ACTIVE[level]:
            if i in self.frozen:
                continue
            if g[i] > 0:
                step = min(delta, 1.0 - u[i])
                s[i] = step
                pred += g[i] * step
            elif g[i] < 0:
                step = min(delta, u[i])
                s[i] = -step
                pred += g[i] * s[i]        # g_i·s_i > 0
        return s, pred

    def _random_step(self, u, delta, level):
        s = np.zeros(DIM)
        for i in ACTIVE[level]:
            if i in self.frozen:
                continue
            s[i] = self.rng.uniform(-delta, delta)
        return s

    def _switching(self, level, delta, stalls):
        """保真度切换规则，返回 (新层级, 新半径, 是否降级)。"""
        cfg = self.cfg
        # 提升：粗层级信任域收敛
        if level == 'L0' and delta < cfg.delta_switch:
            return 'L1', cfg.delta_init, False
        if level == 'L1' and delta < cfg.delta_switch:
            return 'L2', cfg.delta_init, False
        # 额外：粗层级长期停滞也提升，避免低维空间过度沉迷
        if level == 'L0' and stalls >= cfg.stall_accept and delta <= cfg.delta_init:
            return 'L1', cfg.delta_init, False
        if level == 'L1' and stalls >= cfg.stall_accept and delta <= cfg.delta_init:
            return 'L2', cfg.delta_init, False
        # 降级：细层级停滞 → 回 L0 重新探索
        if level in ('L1', 'L2') and delta < cfg.delta_min and stalls >= cfg.stall_accept:
            return 'L0', cfg.delta_init, True
        return None

    def _escape(self, u):
        """降级逃逸：粗层激活维随机重启 + delta/sigma 全范围随机，提供多样性。"""
        uu = u.copy()
        for i in ACTIVE['L0']:
            if i in self.frozen:
                continue
            uu[i] = float(np.clip(uu[i] + self.rng.uniform(-self.cfg.delta_init,
                                                           self.cfg.delta_init), 0.0, 1.0))
        for i in range(N_GLOBAL, DIM):          # delta/sigma 全范围随机
            uu[i] = self.rng.uniform(0.0, 1.0)
        return uu

    # ------------------------------------------------------------------
    def optimize(self, x0, start_level='L0'):
        """从物理 x0 出发优化。返回 (best_x_physical, best_f, best_level)。"""
        cfg = self.cfg
        u = to_unit(np.asarray(x0, dtype=float))
        level = start_level
        delta = cfg.delta_init
        f = self._f(u, level)
        self.best = (u.copy(), f, level)
        stalls = 0

        for it in range(cfg.max_iter):
            g = self._gradient(u, f, level)
            gmax = max(abs(g[i]) for i in ACTIVE[level]) if ACTIVE[level] else 0.0

            if gmax < cfg.flat_grad_eps:
                s = self._random_step(u, delta, level)
                pred = cfg.flat_grad_eps
            else:
                s, pred = self._tr_subproblem(g, u, delta, level)
                if pred < 1e-12:                          # 模型无下降方向
                    s = self._random_step(u, delta, level)
                    pred = cfg.flat_grad_eps

            u_trial = np.clip(u + s, 0.0, 1.0)
            f_trial = self._f(u_trial, level)
            rho = (f_trial - f) / max(pred, 1e-12)
            accepted = rho >= cfg.eta_accept

            if accepted:
                u, f = u_trial, f_trial
                if f > self.best[1]:
                    self.best = (u.copy(), f, level)
                stalls = 0
                if rho >= cfg.eta_expand:
                    delta = min(cfg.delta_max, cfg.gamma_inc * delta)
            else:
                stalls += 1
                delta = cfg.gamma_dec * delta

            self.history.append((level, from_unit(u).copy(), f, delta, rho, accepted))

            # ---- 保真度切换 ----
            sw = self._switching(level, delta, stalls)
            if sw is not None:
                new_level, delta, is_escape = sw
                level = new_level
                stalls = 0
                if is_escape:
                    u = self._escape(u)
                # 在新层级评估当前点：提升时为投影一致点（f 近似不变），
                # 降级时为逃逸点。统一重估一次以保证后续梯度用对层级。
                f = self._f(u, level)

            # 终止：最细层级收敛
            if level == 'L2' and delta < cfg.delta_min:
                break
            if self.evals >= cfg.max_evals:
                break

        best_u, best_f, best_level = self.best
        return from_unit(best_u), best_f, best_level

    def summary(self):
        """打印优化过程摘要。"""
        print('=' * 60)
        print('多保真度信任域优化摘要')
        print('=' * 60)
        print(f'总评估次数 : {self.evals}')
        print(f'历史步数   : {len(self.history)}')
        best_u, best_f, best_level = self.best
        bx = from_unit(best_u)
        print(f'最优层级   : {best_level}')
        print(f'最优效率   : {best_f:.4f}')
        print('最优设计向量（物理单位）：')
        names = ['tower_x', 'tower_y', 'w', 'h', 'H_inst', 'r0', 'K', 'dr', 'ds',
                 'delta0', 'delta1', 'delta2', 'sigma0', 'sigma1', 'sigma2', 'sigma3']
        for n, v in zip(names, bx):
            print(f'  {n:>8s} = {v:8.3f}')


if __name__ == '__main__':
    # 冒烟测试：noon 模式，短预算，验证算法端到端可运行
    from q2_layout import FieldObjective
    obj = FieldObjective(mode='noon')
    cfg = TRConfig(max_iter=20, max_evals=60)
    opt = MultiFidelityTR(obj, cfg, rng=np.random.default_rng(0))

    x0 = np.array([0.0, 0.0, 6.0, 6.0, 4.0, 120.0, 16.0, 15.0, 8.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_x, best_f, best_level = opt.optimize(x0, start_level='L0')
    opt.summary()
