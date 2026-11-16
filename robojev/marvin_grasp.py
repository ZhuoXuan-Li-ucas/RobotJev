#!/usr/bin/env python3
"""点选抓取，天机 Marvin 右臂版（第一步：只用顶部 335，「335 直线接近」那条路，不用腕部 305）。

    conda 环境 laya；cd ~/yuhang_workspace/算法/Jev/RobotJev；命令前加 PYTHONPATH= ~/miniforge3/envs/laya/bin/python
    -m robojev.marvin_grasp --dry                    # 只点选、看判定和规划（SDK 离线算），不连臂
    -m robojev.marvin_grasp --dry --frame f.npz      # 用存下的一帧（npz 里要有 color、depth）
    -m robojev.marvin_grasp --stage plan             # 连臂：回观察位 → 335 重新定位 → 规划，停在观察位
    -m robojev.marvin_grasp --stage approach         # ……再走到预抓取点停下（不合爪）
    -m robojev.marvin_grasp                          # 完整：……直线接近 → 合爪 → 抬 60mm → 335 判成败 → 问要不要放回
    -m robojev.marvin_grasp --force 30               # 只这一次固定合爪力度档位（1~100），不按 CLIP

先要有 335 外参：-m robojev.calibrate_top335_marvin（世界系）。

流程（和旧 click_grasp → run_composite --click 的 ⑧ 分支对应）：
  1. 335 拍一帧，鼠标点物体（窗口操作同 click_grasp）；每个粒度都用 SDK 规划判「能不能抓」
  2. Enter 后才连夹爪、连臂；臂回观察位（竖直向上，不挡 335），按点选的提示在新画面上重新分割、量几何、再规划一次
  3. 夹爪全开 → 关节规划到上方点（必要时经 READY）→ 竖直直线降到预抓取点 → 沿接近方向直线到抓取位
  4. 快合到「比物体宽 12mm」→ 按 CLIP 判的力度 / 速度合上 → 直线抬 60mm → 335 看物体有没有跟着升高 + 夹爪读数
  5. 问：Enter = 放回原处、松开、退回观察位；q = 夹着停在原地

抓取几何（世界系，米）：抓取点 / 支撑面 / 物体点云来自 335（top335.measure_mask，和旧流程同一套）；
开合方向按物体在各水平方向上的宽度从窄到宽试；姿态、逆解、路径全在 robojev.marvin.Planner（SDK）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .marvin import (APPROACH_VEL, FULL_OPEN_M, PRE_GAP_M, Gripper, MarvinArm, Planner, describe, fmt_mm,
                     load_extrinsic)

LIFT_M = 0.06
NEAR_M = 0.12                  # 抓取点周围这么大水平范围里 335 看到的最高点 = 上方点要越过的障碍
MIN_POINTS = 15


class MarvinJudge:
    """点选界面的「能不能抓」（ClickUI 调 judge）。也给重新定位后的规划用。"""

    def __init__(self, planner: Planner, cm335, R, t, full_open=FULL_OPEN_M):
        self.p, self.cm, self.R, self.t, self.full_open = planner, cm335, R, t, full_open
        self.last_plan = None

    def world_geometry(self, obs, plane):
        """335 的测量 → 世界系：抓取点、支撑面 z、抓取高度以上的物体点（米）。"""
        from .top335 import support_flange_z, to_flange, to_flange_many

        c = to_flange(obs.grasp, self.R, self.t)
        sz = support_flange_z(obs, plane, self.R, self.t)
        pts = obs.extra.get("grasp_pts")
        P = None if pts is None or len(pts) < MIN_POINTS else to_flange_many(pts, self.R, self.t)
        return c, sz, P

    @staticmethod
    def width_fn(P):
        if P is None:
            return lambda al: None

        def w(al):
            u = np.array([np.cos(np.radians(al)), np.sin(np.radians(al)), 0.0])
            s = P @ u
            return float(np.percentile(s, 98) - np.percentile(s, 2))
        return w

    def obstacle_top(self, depth, c):
        from .top335 import scene_obstacle_z

        f = np.array([c[0], c[1], 5.0])          # 起点 z 给大：观察位时臂不在画面里，不用排除臂自己
        z, n = scene_obstacle_z(depth, self.cm, self.R, self.t, f, f, radius=NEAR_M, body=0.0)
        return z

    def plan(self, obs, plane):
        """-> (plan dict 或 None, 说明)"""
        h = obs.extra.get("height")
        if h is None:
            return None, "量不到它下面的支撑面（不是放在平面上的东西，或者底部看不见），测不出高度"
        c, sz, P = self.world_geometry(obs, plane)
        if sz is None:
            return None, "量不到支撑面高度"
        top = self.obstacle_top(obs.extra.get("scene_depth"), c) if obs.extra.get("scene_depth") is not None else None
        try:
            p = self.p.plan_grasp(c, sz, self.width_fn(P), self.full_open, obstacle_top=top)
        except RuntimeError as e:
            return None, str(e)
        p["height"] = h
        return p, describe(p)

    def judge(self, tf, depth, mask):
        from .top335 import measure_mask

        r = measure_mask(tf, depth, mask, self.cm, self.R, self.t)
        if r is None:
            return False, "这块区域没有有效深度，量不出几何", None
        obs, plane = r
        p, why = self.plan(obs, plane)
        if p is not None:
            obs.extra["width"] = p["width"]
        self.last_plan = p
        return p is not None, ("能抓：" + why) if p is not None else why, obs


def close_lift_judge(arm, g, plan, grip_plan, follows, force_override=None):
    """快合到比物体宽 PRE_GAP → 按力度合上 → 直线抬 LIFT → 335 看物体有没有跟着走 + 夹爪读数。返回成没成功。"""
    width = plan["width"] if plan["width"] is not None else 0.0
    expect = g.for_width(width)
    g.to(g.for_width(width + PRE_GAP_M), wait=1.2, speed=1000)
    F = force_override if force_override is not None else grip_plan.force
    S = grip_plan.speed
    v = g.to(1.0, wait=2.5 if S <= 300 else 2.0, force=F, speed=S)
    print(f"   合爪：力 {F} 档、速度 {S}，读数 {v:.2f}（按宽度 {width * 1000:.0f}mm 应读 {expect:.2f}）"
          + ("，合到底了 —— 两指之间没东西" if v >= 0.97 else ""))
    T_lift = plan["T_grasp"].copy()
    T_lift[2, 3] += LIFT_M
    arm.move_line(T_lift, f"抬 {LIFT_M * 1000:.0f}mm")
    time.sleep(0.3)
    ok, txt = follows(LIFT_M)
    v2 = g.get()
    if ok is None:
        ok, txt = v2 < 0.97, txt + f"；抬起后夹爪读数 {v2:.2f}"
    grip_ok = v < 0.97 and abs(v - expect) < 0.2 and v2 < 0.97
    res = bool(grip_ok and ok)
    print(f"结果：夹爪 {v:.2f}→{v2:.2f} {'✓' if grip_ok else '✗'}；{txt} {'✓' if ok else '✗'}  -> {'成功' if res else '失败'}")
    return res, T_lift


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="只点选、看判定和规划，不连臂")
    ap.add_argument("--frame", default=None, help="（配 --dry）用存下的 npz（color, depth）代替现拍")
    ap.add_argument("--stage", choices=["plan", "approach", "grasp"], default="grasp")
    ap.add_argument("--force", type=int, default=None, help="只这一次：合爪力度档位 1~100，覆盖 CLIP")
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings("ignore")

    from .click_grasp import grab, select
    from .objects import CameraModel, TargetFinder

    R, t, meta = load_extrinsic()
    print(f"335 外参：{meta['date']}，留一误差中位 {meta['loo_median_mm']:.1f}mm")
    cm335 = CameraModel(Path(__file__).resolve().parent / "calibration" / "orbbec335_params.json")
    planner = Planner()
    judge = MarvinJudge(planner, cm335, R, t)
    tf = TargetFinder("pick up the object")

    if a.dry and a.frame:
        d = np.load(a.frame)
        sel = select(tf, d["color"], d["depth"], judge)
        print("没选" if sel is None else f"选中「{sel['name']}」：{sel['verdict']}（--dry，不抓）")
        return 0

    from .camera import TOP_SERIAL, OrbbecRGBD
    cam = OrbbecRGBD(serial=TOP_SERIAL)
    cam.connect()
    try:
        color, depth = grab(cam)
        print("点选窗口（335 这一帧）：左键点物体，Tab 换粒度，Enter 抓，Esc 退出")
        sel = select(tf, color, depth, judge)
        if sel is None:
            print("没选，退出")
            return 0
        print(f"点选：「{sel['name']}」-> {sel['verdict']}")
        if a.dry:
            return 0
        tf.set_instruction(f"pick up the {sel['name']}")
        return run(a, tf, cam, cm335, R, t, judge, planner, sel)
    finally:
        cam.disconnect()


def run(a, tf, cam, cm335, R, t, judge, planner, sel) -> int:
    from .grip_force import judge as clip_judge
    from .run_composite import TopFollows
    from .top335 import locate_click, to_flange

    with Gripper() as g, MarvinArm(planner) as arm:
        input("要动臂了：先回观察位（竖直向上）。手放在急停上，按 Enter 继续 > ")
        arm.go_home()
        time.sleep(0.5)
        r = locate_click(tf, cam, cm335, R, t, sel, np.eye(3))
        if r is None:
            return 1
        obs, plane = r
        p, why = judge.plan(obs, plane)
        if p is None:
            print(f"重新定位后抓不了：{why}")
            return 1
        w_txt = "—" if p["width"] is None else f"{p['width'] * 1000:.0f}mm"
        print(f"重新定位：抓取点世界 {fmt_mm(p['c'])}mm，支撑面 z={p['support_z'] * 1000:.0f}mm，"
              f"物体高 {p['height'] * 1000:.0f}mm，沿开合方向宽 {w_txt}")
        print(f"规划：{why}")
        if a.stage == "plan":
            return 0
        grip_plan = clip_judge(tf, cam.read().color, obs.xyxy, tf.phrase)
        print(f"CLIP：判为「{grip_plan.kind}」-> 力 {grip_plan.force}、速度 {grip_plan.speed}"
              + (f"（--force 改成 {a.force}）" if a.force is not None else ""))
        follows = TopFollows(tf, cam, cm335, R, t, before=to_flange(obs.surface, R, t), click=True)
        follows.px0 = obs.mask_px

        g.to(0.0, wait=1.5, speed=1000)
        if p["via"] is not None:
            arm.move_joints(p["via"], "经 READY")
        arm.move_joints(p["q_above"], "到上方点")
        arm.move_line(p["T_pre"], "降到预抓取点")
        if a.stage == "approach":
            print("停在预抓取点（--stage approach）。回观察位：")
            arm.move_line(p["T_above"], "回上方点")
            arm.go_home()
            return 0
        arm.move_line(p["T_grasp"], "直线接近", vel=APPROACH_VEL, floor_z=p["support_z"])
        ok, T_lift = close_lift_judge(arm, g, p, grip_plan, follows, a.force)
        ans = input("Enter = 放回原处、松开、回观察位；q = 夹着停在这里 > ").strip().lower()
        if ans == "q":
            return 0 if ok else 1
        arm.move_line(p["T_grasp"], "放回", vel=APPROACH_VEL, floor_z=p["support_z"])
        g.to(0.0, wait=1.5, speed=1000)
        arm.move_line(p["T_pre"], "退回预抓取点", vel=APPROACH_VEL)
        arm.move_line(p["T_above"], "回上方点")
        arm.go_home()
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
