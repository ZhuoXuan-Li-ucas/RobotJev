"""天机 Marvin 臂层的离线测试：不连控制器、不开相机、不加载 YOLO/CLIP/SAM。

要 天机Marvin/ 下的官方 SDK（libKine）和 world_calib/B_T_world_base.json。
  1. 观察位 / READY 的正解和世界系定义对得上（READY 时夹爪是平的：工具 X 水平）
  2. SDK 规划：桌上乒乓球在抓取区几个位置都规划得出，倾角 45° 或 60°，指尖不低于支撑面
  3. 335 外参拟合：已知 R、t、球心偏移 o，加 1mm 噪声，能解回来
  4. 合成的 335 画面（桌面 + 圆柱）走 measure_mask → MarvinJudge.plan，量出的高度、位置对、规划得出
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from robojev.marvin import HOME_Q, READY_Q, TIP_CLEAR_M, TIP_M, Planner  # noqa: E402

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name if cond else f"{name}: {detail}")


pl = Planner()

# 1. 正解
T = pl.tcp(READY_Q)
ok("READY TCP 世界位置", np.allclose(T[:3, 3], [0.1745, 0.527, 0.305], atol=0.002), T[:3, 3])
ok("READY 工具 X 水平（夹爪是平的）", abs(T[2, 0]) < 1e-3, T[:3, 0])
ok("观察位竖直向上", np.allclose(pl.tcp(HOME_Q)[:3, 3], [0.1745, 0.0, 0.796], atol=0.002), pl.tcp(HOME_Q)[:3, 3])
ok("观察位 → READY 关节路径不碰", pl.joint_path([HOME_Q, READY_Q]) is None)

# 2. 规划
for xy in [(0.10, 0.45), (0.25, 0.55), (-0.10, 0.50), (0.30, 0.70)]:
    c = np.array([xy[0], xy[1], 0.02])
    try:
        p = pl.plan_grasp(c, 0.0, lambda al: 0.040, obstacle_top=0.04)
    except RuntimeError as e:
        ok(f"乒乓球 {xy} 规划", False, e)
        continue
    Tg = p["T_grasp"]
    tip = Tg[:3, 3] + TIP_M * Tg[:3, 2]
    ok(f"乒乓球 {xy} 倾角", p["theta"] in (45.0, 60.0), p["theta"])
    ok(f"乒乓球 {xy} 指尖离桌", tip[2] >= TIP_CLEAR_M - 1e-6, tip)
    ok(f"乒乓球 {xy} 开合方向水平", abs(Tg[2, 0]) < 1e-6, Tg[:3, 0])
    ok(f"乒乓球 {xy} 抓取位 FK 回代", np.linalg.norm(pl.tcp(p["q_grasp"])[:3, 3] - Tg[:3, 3]) < 1e-3)
try:
    pl.plan_grasp(np.array([0.10, 0.50, 0.02]), 0.0, lambda al: 0.095)
    ok("太宽的物体判抓不了", False, "居然规划出来了")
except RuntimeError as e:
    ok("太宽的物体判抓不了", "放不进" in str(e), e)

# 3. 外参拟合
from robojev.calibrate_top335_marvin import fit, plan as calib_plan  # noqa: E402

Ts = [pl.tcp(d[2]) for _, d, _ in calib_plan(pl, HOME_Q) if d]
ok("标定走点数", len(Ts) >= 12, len(Ts))
rng = np.random.default_rng(1)
ax = rng.normal(size=3)
ax /= np.linalg.norm(ax)
K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
R0 = np.eye(3) + np.sin(2.2) * K + (1 - np.cos(2.2)) * K @ K
t0, o0 = np.array([0.3, -0.2, 1.4]), np.array([0.004, -0.006, 0.012])
C = np.array([R0 @ (T[:3, 3] + T[:3, :3] @ o0) + t0 for T in Ts]) + rng.normal(scale=0.001, size=(len(Ts), 3))
R1, t1, o1, e = fit(Ts, C)
ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
ok("外参转角", ang < 0.3, ang)
ok("外参平移", np.linalg.norm(t1 - t0) < 0.002, t1 - t0)
ok("球心偏移 o", np.linalg.norm(o1 - o0) < 0.002, o1)

# 4. 合成 335 画面
from robojev.marvin_grasp import MarvinJudge  # noqa: E402
from robojev.objects import CameraModel, TargetFinder  # noqa: E402
from robojev.top335 import measure_mask  # noqa: E402

cm = CameraModel(Path(__file__).resolve().parents[1] / "robojev" / "calibration" / "orbbec335_params.json")
pos, look = np.array([0.1, 1.5, 1.0]), np.array([0.1, 0.5, 0.0])
zc = (look - pos) / np.linalg.norm(look - pos)
up = np.array([0, 0, 1.0])
yc = -(up - (up @ zc) * zc)
yc /= np.linalg.norm(yc)
R = np.column_stack([np.cross(yc, zc), yc, zc]).T
t = -R @ pos


def render(cx, cy, r=0.03, h=0.08):
    X, Y = np.meshgrid(np.arange(-0.4, 0.6, 0.002), np.arange(0.4, 1.0, 0.002))
    desk = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    desk = desk[(desk[:, 0] - cx) ** 2 + (desk[:, 1] - cy) ** 2 > r * r]
    A, Z = np.meshgrid(np.arange(0, 2 * np.pi, 0.01), np.arange(0, h, 0.0015))
    side = np.c_[cx + r * np.cos(A.ravel()), cy + r * np.sin(A.ravel()), Z.ravel()]
    A, RR = np.meshgrid(np.arange(0, 2 * np.pi, 0.02), np.arange(0, r, 0.0015))
    top = np.c_[cx + RR.ravel() * np.cos(A.ravel()), cy + RR.ravel() * np.sin(A.ravel()), np.full(A.size, h)]
    P = np.vstack([desk, side, top])
    lab = np.r_[np.zeros(len(desk)), np.ones(len(side) + len(top))].astype(bool)
    Pc = P @ R.T + t
    uv = np.round(cm.project(Pc)).astype(int)
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480) & (Pc[:, 2] > 0)
    uv, zz, lab = uv[keep], Pc[keep, 2], lab[keep]
    order = np.argsort(-zz)                     # 远的先写、近的后写覆盖 = z-buffer
    depth = np.zeros((480, 640), np.uint16)
    mask = np.zeros((480, 640), bool)
    depth[uv[order, 1], uv[order, 0]] = (zz[order] * 1000).astype(np.uint16)
    mask[uv[order, 1], uv[order, 0]] = lab[order]
    return depth, mask


tf = TargetFinder.__new__(TargetFinder)          # 只用几何，不加载模型
tf.cam, tf.last, tf.phrase, tf.up = cm, None, "object", None
J = MarvinJudge(pl, cm, R, t)
for cx, cy in [(0.10, 0.52), (0.25, 0.65)]:
    depth, mask = render(cx, cy)
    r = measure_mask(tf, depth, mask, cm, R, t)
    if r is None:
        ok(f"圆柱 {(cx, cy)} 量几何", False, "measure_mask None")
        continue
    obs, plane = r
    c, sz, P = J.world_geometry(obs, plane)
    ok(f"圆柱 {(cx, cy)} 高度", obs.extra["height"] is not None and abs(obs.extra["height"] - 0.08) < 0.006,
       obs.extra["height"])
    ok(f"圆柱 {(cx, cy)} 支撑面", sz is not None and abs(sz) < 0.005, sz)
    ok(f"圆柱 {(cx, cy)} 抓取点水平位置", np.linalg.norm(c[:2] - [cx, cy]) < 0.012, c)
    p, why = J.plan(obs, plane)
    ok(f"圆柱 {(cx, cy)} 规划", p is not None, why)

print(f"{len(PASS)} passed, {len(FAIL)} failed")
for f in FAIL:
    print("FAIL", f)
sys.exit(1 if FAIL else 0)
