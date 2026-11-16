#!/usr/bin/env python3
"""俯视 Gemini 335 ↔ 世界系 外参标定（天机 Marvin 右臂）：夹着乒乓球走一组点，335 看球心，拟合。

    cd ~/yuhang_workspace/算法/Jev/RobotJev
    PYTHONPATH= ~/miniforge3/envs/laya/bin/python -m robojev.calibrate_top335_marvin --dry-run   # 只算点位 / 逆解 / 路径，不连臂、不开相机
    PYTHONPATH= ~/miniforge3/envs/laya/bin/python -m robojev.calibrate_top335_marvin             # 真跑（手放在急停上）
    PYTHONPATH= ~/miniforge3/envs/laya/bin/python -m robojev.calibrate_top335_marvin --refit     # 不动臂，用上次存的点重新拟合

真跑的顺序：开 335 → 连夹爪、连 B 臂（只连，不动）→ 夹爪张开，你把乒乓球放进两指之间，Enter → 小力合上
→ 再按一次 Enter 才开始动：回观察位（竖直向上）→ 逐点走（每点停稳后 335 看球）→ 回观察位（球还夹着）→ 拟合。

拟合什么：335 系里的球心 c = R·(TCP 位置 + TCP 姿态·o) + t（米）
  R, t：世界 → 335（结果写 calibration/top335_marvin.json，形式和旧 top335_extrinsic 一样，top335 的函数原样能用）
  o   ：球心在工具系里相对 TCP（夹持中心）的偏移 —— 球放得不正也没关系，一起解出来；点位换了几种倾角 / 倾斜方向，o 才解得出。
做法：先当 o = 0 用 Kabsch，然后「给 R 解 (o, t) 的线性最小二乘」和「给 o 做 Kabsch」交替到收敛。
质量关和旧版一样：球半径用已知值（自由半径偏 >30% 的点丢掉），逐个去掉最差的点，留一误差中位 > 5mm 不写。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .marvin import (CALIB, EXTRINSIC, HOME_Q, READY_Q, SHOULDER_W, Gripper, MarvinArm, Planner, fmt_mm,
                     grasp_R)

POINTS = CALIB / "top335_marvin_points.npz"
GRID_X = (-0.10, 0.10, 0.30)
GRID_Y = (0.45, 0.54, 0.64)
GRID_Z = (0.12, 0.22)            # TCP 世界 z：球底离桌面 ≥ 10cm（11-16 离线：18 点里 16 点能走）
ORIENTS = ((45.0, 0.0), (30.0, 30.0), (60.0, -30.0), (45.0, -40.0), (30.0, -20.0), (60.0, 30.0))
#           (倾角 θ, 倾斜方向相对「肩 → 点」的偏转)；逐点轮换，让 o 解得出来
BALL_R = 0.020
MAX_LOO_M = 0.005
GRIP_FORCE = 20


def targets():
    """蛇形走点；每个点给一串候选姿态（第一个够不着就试下一个）。"""
    out = []
    k = 0
    for iz, z in enumerate(GRID_Z):
        layer = []
        for j, y in enumerate(GRID_Y):
            xs = GRID_X if j % 2 == 0 else GRID_X[::-1]
            layer += [(x, y) for x in xs]
        if iz % 2:
            layer = layer[::-1]
        for x, y in layer:
            p = np.array([x, y, z])
            az = float(np.degrees(np.arctan2(y - SHOULDER_W[1] / 1000, x - SHOULDER_W[0] / 1000)))
            cands = []
            for i in range(len(ORIENTS)):
                th, dl = ORIENTS[(k + i) % len(ORIENTS)]
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = grasp_R(th, az + dl, 0.0), p
                cands.append((th, dl, T))
            out.append((p, cands))
            k += 1
    return out


def plan(pl: Planner, q_start):
    """逐点：候选姿态依次试 SDK 逆解（参考角 = 上一点），再查从上一点过来的关节插值路径。"""
    steps, q_prev = [], np.asarray(q_start, float)
    for p, cands in targets():
        done = None
        whys = []
        for th, dl, T in cands:
            refs = [READY_Q] if np.abs(q_prev[3]) < 1e-3 else [q_prev, READY_Q]
            sols = [pl.solve(T, r) for r in refs]
            ok = [s for s, _ in sols if s is not None]
            if not ok:
                whys.append(f"θ{th:.0f}/{dl:+.0f}: {sols[0][1]}")
                continue
            q = min(ok, key=lambda s: np.abs(s - q_prev).max())      # 离上一点近的，少换构型
            via = None
            path = pl.joint_path([q_prev, q])
            if path and pl.joint_path([q_prev, READY_Q, q]) is None:
                path, via = None, READY_Q
            if path:
                whys.append(f"θ{th:.0f}/{dl:+.0f}: {path}")
                continue
            done = (th, dl, q, via)
            break
        steps.append((p, done, whys))
        if done:
            q_prev = done[2]
    return steps


# ---------------------------------------------------------------- 拟合

def _kabsch(A, B):
    """B ≈ R A + t。"""
    am, bm = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - am).T @ (B - bm))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, bm - R @ am


def fit(Ts, C, iters=50):
    """Ts：每点 TCP 位姿（世界，4x4 米）；C：335 系球心。返回 (R, t, o, 每点残差)。"""
    P = np.array([T[:3, 3] for T in Ts])
    Rs = [T[:3, :3] for T in Ts]
    o = np.zeros(3)
    R, t = _kabsch(P, C)
    for _ in range(iters):
        A = np.vstack([np.c_[R @ Ri, np.eye(3)] for Ri in Rs])          # c − R p = R Ri o + t
        b = (C - P @ R.T).ravel()
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        o_new = sol[:3]
        R, t = _kabsch(P + np.array([Ri @ o_new for Ri in Rs]), C)
        if np.linalg.norm(o_new - o) < 1e-7:
            o = o_new
            break
        o = o_new
    W = P + np.array([Ri @ o for Ri in Rs])
    err = np.linalg.norm(C - (W @ R.T + t), axis=1)
    return R, t, o, err


def fit_and_save(Ts, clouds, r_ball=BALL_R) -> int:
    from .calibrate_top335 import sphere_fit

    np.savez(POINTS, T=np.array(Ts), **{f"P{k}": P for k, P in enumerate(clouds)})
    free = [sphere_fit(P) for P in clouds]
    rf = np.array([r for _, r in free])
    good = np.abs(rf / r_ball - 1.0) < 0.3
    print(f"各点自由拟合半径 {np.round(rf * 1000, 1).tolist()} mm；已知 {r_ball * 1000:.1f}mm，偏离 <30% 的 {good.sum()}/{len(rf)} 点可用")
    if good.sum() < 6:
        print("可用点不到 6 个，不写外参")
        return 1
    C = np.array([sphere_fit(P, r=r_ball, c0=c)[0] for P, (c, _) in zip(clouds, free)])
    Ts = np.array(Ts)
    keep = good.copy()
    while True:
        idx = np.nonzero(keep)[0]
        R, t, o, e = fit(Ts[idx], C[idx])
        W = Ts[:, :3, 3] + np.einsum("nij,j->ni", Ts[:, :3, :3], o)
        err = np.linalg.norm(C - (W @ R.T + t), axis=1)
        worst = int(np.argmax(np.where(keep, err, -1)))
        if keep.sum() <= 6 or err[worst] < max(0.005, 3 * np.median(err[keep])):
            break
        keep[worst] = False
    idx = np.nonzero(keep)[0]
    loo = []
    for i in idx:
        m = idx[idx != i]
        Ri, ti, oi, _ = fit(Ts[m], C[m])
        loo.append(np.linalg.norm(C[i] - (Ri @ (Ts[i, :3, 3] + Ts[i, :3, :3] @ oi) + ti)))
    print(f"拟合残差：中位 {np.median(err[keep]) * 1000:.1f}mm  最大 {err[keep].max() * 1000:.1f}mm（用 {keep.sum()}/{len(err)} 点）")
    print(f"留一预测误差：中位 {np.median(loo) * 1000:.1f}mm  最大 {max(loo) * 1000:.1f}mm")
    print(f"球心相对 TCP（工具系）偏移 o = {np.round(o * 1000, 1).tolist()} mm（夹得正应接近 0；Z 是接近方向）")
    print(f"335 在世界系的位置 {fmt_mm(-R.T @ t)} mm，光轴（相机 +Z）在世界系 {np.round(R.T[:, 2], 3).tolist()}")
    out = dict(R=R.tolist(), t=t.tolist(), ball_offset_tool_m=o.tolist(), frame="world",
               rms_mm=float(np.sqrt(np.mean(err[keep] ** 2)) * 1000), loo_median_mm=float(np.median(loo) * 1000),
               loo_max_mm=float(max(loo) * 1000), ball_radius_m=r_ball,
               points=[dict(T=T.tolist(), ball=c.tolist(), used=bool(u)) for T, c, u in zip(Ts, C, keep)],
               note="335 系点 = R·世界点 + t（米）；世界系见 天机Marvin/README；夹乒乓球标的，球心偏移 o 已解出（不在 t 里）",
               date=time.strftime("%Y-%m-%d %H:%M"))
    if np.median(loo) > MAX_LOO_M:
        bad = EXTRINSIC.with_name(EXTRINSIC.stem + "_rejected.json")
        bad.write_text(json.dumps(out, indent=1))
        print(f"留一误差中位 {np.median(loo) * 1000:.1f}mm > {MAX_LOO_M * 1000:.0f}mm，不采用；写到 {bad.name} 备查")
        return 1
    EXTRINSIC.write_text(json.dumps(out, indent=1))
    print(f"已写 {EXTRINSIC}")
    return 0


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="只算点位、逆解、关节路径，不连臂不开相机")
    ap.add_argument("--refit", action="store_true", help="不动臂，用上次存的点重新拟合")
    ap.add_argument("--radius", type=float, default=BALL_R)
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")

    if a.refit:
        d = np.load(POINTS)
        n = len(d["T"])
        return fit_and_save(list(d["T"]), [d[f"P{k}"] for k in range(n)], a.radius)

    pl = Planner()
    if a.dry_run:
        steps = plan(pl, HOME_Q)
        n_ok = 0
        for k, (p, done, whys) in enumerate(steps):
            if done:
                n_ok += 1
                th, dl, q, via = done
                print(f"[{k:2d}] TCP {fmt_mm(p)}mm  倾 {th:.0f}°/{dl:+.0f}°  q={np.round(q, 1).tolist()}"
                      + ("  （经 READY）" if via is not None else ""))
            else:
                print(f"[{k:2d}] TCP {fmt_mm(p)}mm  跳过：{'；'.join(whys[:2])}")
        q_last = next((d[2] for _, d, _ in reversed(steps) if d), HOME_Q)
        back = pl.joint_path([q_last, HOME_Q])
        if back and pl.joint_path([q_last, READY_Q, HOME_Q]) is None:
            back = None
        print(f"{n_ok}/{len(steps)} 个点能走；最后回观察位：{'可以（必要时经 READY）' if back is None else back}"
              "（335 能不能看见球只有真跑才知道，看不见的点自动跳过）")
        return 0

    from .calibrate_top335 import ball_points, sphere_fit
    from .camera import TOP_SERIAL, OrbbecRGBD
    from .objects import CameraModel, TargetFinder

    tf = TargetFinder("pick up the ping pong ball")
    tf.cam = CameraModel(CALIB / "orbbec335_params.json")
    cam = OrbbecRGBD(serial=TOP_SERIAL)
    cam.connect()
    Ts, clouds = [], []
    try:
        with Gripper(force=GRIP_FORCE, speed=500) as g, MarvinArm(pl) as arm:
            v = g.get()
            if not 0.2 < v < 0.97:
                g.to(0.0, wait=2.0)
                input("夹爪已张开。把乒乓球放进两指之间（尽量在指尖往回 2~3cm 处），扶着，按 Enter 合爪 > ")
                v = g.to(1.0, wait=2.5, force=GRIP_FORCE, speed=500)
            if not 0.2 < v < 0.97:
                raise SystemExit(f"夹爪读数 {v:.2f}，不像夹着球（应 ~0.6），不走")
            print(f"夹着球（读数 {v:.2f}）")
            steps = plan(pl, HOME_Q)
            n_ok = sum(1 for _, d, _ in steps if d)
            input(f"要走 {n_ok} 个点：先回观察位（竖直向上），再逐点走，最后回观察位。手放在急停上，按 Enter 开始 > ")
            arm.go_home()
            for k, (p, done, whys) in enumerate(steps):
                if not done:
                    print(f"[{k:2d}] 跳过 {fmt_mm(p)}mm：{whys[0] if whys else ''}")
                    continue
                th, dl, q, via = done
                if via is not None:
                    arm.move_joints(via, f"点 {k}（经 READY）")
                arm.move_joints(q, f"点 {k}")
                time.sleep(0.6)
                T = pl.tcp(arm.q())
                P = ball_points(tf, cam)
                if P is None:
                    print(f"[{k:2d}] TCP {fmt_mm(T[:3, 3])}mm  335 里没认出球")
                    continue
                c, r = sphere_fit(P)
                Ts.append(T)
                clouds.append(P)
                print(f"[{k:2d}] TCP {fmt_mm(T[:3, 3])}mm 倾 {th:.0f}°  球心(335) {fmt_mm(c)}mm 半径 {r * 1000:.1f}mm "
                      f"{len(P)} 点  夹爪 {g.get():.2f}")
            arm.go_home()
    finally:
        cam.disconnect()
    if len(Ts) < 6:
        raise SystemExit(f"有效点只有 {len(Ts)} 个（至少 6 个），没写结果")
    return fit_and_save(Ts, clouds, a.radius)


if __name__ == "__main__":
    raise SystemExit(main())
