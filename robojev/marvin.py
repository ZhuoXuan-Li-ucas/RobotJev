"""天机 Marvin 右臂（B）+ HKVR 夹爪：robojev 的臂层。2026-11-16 起取代 AR5（atom_robot.ar5）。

坐标：对外一律**世界系、米**（天机Marvin/README「世界坐标系」：原点 = B 臂底座安装面中心，+X 右、+Y 前、+Z 上，
桌面 z≈0 和肩同高）。SDK（libKine / 控制器）里是 mm、度，只在本文件里换算。
TCP = 夹持中心：控制器里设好的工具（法兰 +Z 100 mm、绕 Z 转 45°），工具 X = 手指开合方向，工具 Z = 接近方向，
指尖在 TCP 前 50 mm。

运动学全走官方 SDK：逆解 ik / ik_nsp（扫臂角）、直线 movLA、关节规划 setPln_joint、直线执行 setPln_Cart
（调用顺序照 SDK 的 AI_GUIDE 7.2/7.3 和 showcase_pln_cart_positionMode.py）。
自己写的只有「这个解能不能执行」的检查：SDK 运行限位余量、肘 / 腕 / 法兰离桌、离机身、指尖离支撑面。

抓取姿态（11-16 用户定）：夹爪不能竖直朝下（桌面和肩同高，J6 只能弯 ±60°，marvin_reach.py 扫过），
斜 45° 抓，够不着换 60°；开合方向保持水平，倾斜方向 ⟂ 开合方向。

安全（天机Marvin/README）：只碰 B 臂；不调 check_error_and_clear（它会先清 A 臂）；B 臂有错就停，不自动清；
每次运动都限速、等到位、盯关节包络（跑出包络就 stopRunPln + soft_stop）；退出一定 release。
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[4]
MARVIN_DIR = WS / "数据_硬件" / "天机Marvin"
SO101 = WS / "算法" / "DexGraspVLA-" / "SO101"
for _p in (MARVIN_DIR, SO101):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from marvin_common import DEFAULT_IP, DEFAULT_SDK, Marvin, fk_points  # noqa: E402
from marvin_reach import SHOULDER_W, candidates, grasp_R  # noqa: E402
from marvin_world_calib import KINE_CFG, Kine, T_to_xyzabc, load_T, xyzabc_to_T  # noqa: E402

CALIB = Path(__file__).resolve().parent / "calibration"
EXTRINSIC = CALIB / "top335_marvin.json"

ARM = "B"
TOOL = [0.0, 0.0, 100.0, 0.0, 0.0, 45.0]          # 相对法兰 mm/度；连上后核对控制器里的是不是这个
HOME_Q = np.array([90.0, -90.0, 0.0, 0.0, 0.0, 0.0, 0.0])        # 观察位：竖直向上，不挡 335（11-16 用户定）
READY_Q = np.array([90.0, -90.0, -90.0, -90.0, 45.0, 0.0, 0.0])  # 11-14 停过的姿态，TCP 世界 (0.17, 0.53, 0.31) 朝前；
                                                                 # 逆解参考角（HOME 的 J4=0 是奇异位，SDK 不收）+ 关节路径中转点
TILTS = (45.0, 60.0)          # 先 45°，够不着换 60°
TIP_M = 0.050                 # 指尖在 TCP 前多远（用户：指尖离法兰 150 mm）
PAD_M = 0.045                 # 从指尖往回能夹东西的长度（用户：40~50 mm）
TIP_CLEAR_M = 0.008           # 指尖离支撑面至少这么高
DESK_Z_M = 0.0                # 桌面世界 z（用户 11-16：粗估约 0）
DESK_FRONT_Y_M = 0.40         # 桌子前沿（用户：最近 40 cm）；更靠近机身的部分下面没有桌子
LINK_CLEAR_M = 0.05           # 肘 / 腕 / 法兰中心离桌面至少这么高（关节壳半径）
BODY_Y_M, BODY_X_M = 0.15, -0.05   # 未实测的机身禁区：y < 0.15 且 x < −0.05（原点是右臂安装面，机身在 −X 侧）
MARGIN_DEG = 5.0              # 离 SDK 运行限位至少多少度（含 J6/J7 耦合）
PRE_BACKS = (0.10, 0.08, 0.06)     # 预抓取点：沿接近方向往回退，够不着就缩短
TRANSIT_Z_M = 0.20            # 预抓取点上方的「上方点」首选高度（TCP 世界 z）
CLEAR_M = 0.08                # 上方点比 335 看到的走廊里最高的东西高这么多
PRE_GAP_M = 0.012             # 夹得下：宽度 ≤ 满开 − 这个
FULL_OPEN_M = 0.098           # TG9801 手册行程

JOINT_VEL, JOINT_ACC = 0.08, 0.10       # setPln_joint 的速度 / 加速度比例（0~1，相对关节最大速度 180°/s）
LINE_VEL, LINE_ACC = 30.0, 60.0         # movLA 直线 mm/s、mm/s²
APPROACH_VEL = 15.0                     # 预抓取 → 抓取那一段
POS_MODE_PCT = 20                       # 位置模式的 set_vel_acc 百分比（规划点集按自己的速度走，这个只是跟随上限）
GUARD_DEG = 3.0                         # 关节跑出「规划包络 ± 这个」就停


def _m2mm(T):
    T = np.array(T, float).copy()
    T[:3, 3] *= 1000.0
    return T


def _mm2m(T):
    T = np.array(T, float).copy()
    T[:3, 3] /= 1000.0
    return T


def fmt_mm(p):
    return np.round(np.asarray(p, float) * 1000).astype(int).tolist()


_LIBC = ctypes.CDLL(None)


@contextlib.contextmanager
def quiet():
    """SDK 每次规划都往 stdout 打「Plan MOVLA successful」（fx_kine.py 的 print）和「Joint N exceed limit」
    （libKine 的 printf），点选时一个粒度要规划几十次会刷屏。只在 SDK 计算调用期间把 fd 1 接到 /dev/null。"""
    sys.stdout.flush()
    saved = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        _LIBC.fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
        os.close(null)


# ================================================================ 规划（只算，不连控制器）

class Planner:
    """SDK 逆解 / 直线规划 + 能不能执行的检查。全部离线，点选界面判「能不能抓」也用它。"""

    def __init__(self, desk_z: float = DESK_Z_M):
        self.k = Kine(ARM)
        self.T_wb = load_T(ARM)                     # mm
        self.T_bw = np.linalg.inv(self.T_wb)
        self.dh = np.array(self.k.dh, float)
        self.lim = np.array(self.k.pnva, float)[:, :2]   # 每行 [正限, 负限]
        self.desk_z = desk_z

    # ---- 正解 / 几何
    def tcp(self, q) -> np.ndarray:
        """世界系 TCP 位姿（4x4，米）。"""
        return _mm2m(self.T_wb @ self.k.tcp(np.asarray(q, float), TOOL))

    def links(self, q) -> np.ndarray:
        """肘（J3）往后各关节原点 + 法兰，世界系，米。"""
        p = fk_points(self.dh[:7], self.dh[7], np.asarray(q, float))[2:]
        return (p @ self.T_wb[:3, :3].T + self.T_wb[:3, 3]) / 1000.0

    def check_q(self, q, floor_z=None):
        """None = 这个构型不碰桌、不碰机身、指尖不戳进支撑面；否则返回原因。"""
        L = self.links(q)
        over = L[L[:, 1] > DESK_FRONT_Y_M]
        if len(over) and over[:, 2].min() < self.desk_z + LINK_CLEAR_M:
            return f"肘/腕/法兰离桌面只有 {(over[:, 2].min() - self.desk_z) * 1000:.0f}mm"
        if np.any((L[:, 1] < BODY_Y_M) & (L[:, 0] < BODY_X_M)):
            return "肘/腕伸到机身前面"
        T = self.tcp(q)
        tip = T[:3, 3] + TIP_M * T[:3, 2]
        fz = self.desk_z if floor_z is None else max(self.desk_z, floor_z)
        if tip[1] > DESK_FRONT_Y_M and tip[2] < fz + TIP_CLEAR_M - 1e-4:
            return f"指尖离支撑面只有 {(tip[2] - fz) * 1000:.0f}mm"
        return None

    @staticmethod
    def _margin(r) -> float:
        return float(np.minimum(r["run_lmt_p"] - r["q"], r["q"] - r["run_lmt_n"]).min())

    # ---- 逆解
    def solve(self, T_world, ref, floor_z=None):
        """世界系 TCP 位姿（米）→ (q 度, None) 或 (None, 原因)。先 SDK ik（离 ref 最近的解），
        不合格（超限 / 奇异 / 离限位 < MARGIN / 碰桌碰机身）再 SDK ik_nsp 扫臂角，挑合格且离 ref 最近的。"""
        ref = np.asarray(ref, float)
        Tb = self.T_bw @ _m2mm(T_world)
        with quiet():
            r = self.k.ik(Tb, ref, TOOL)

        def good(g):
            return g["valid"] and self._margin(g) >= MARGIN_DEG and self.check_q(g["q"], floor_z) is None

        if r["ok"] and good(r):
            return r["q"], None
        with quiet():
            scan = self.k.ik_nsp_scan(Tb, ref, TOOL, step=10.0)
        cand = [g for _, g in scan if good(g)]
        if cand:
            return min(cand, key=lambda g: np.abs(g["q"] - ref).max())["q"], None
        if not r["ok"] or r["out_of_range"]:
            return None, "够不着（SDK ik 无解）"
        if not r["valid"] or self._margin(r) < MARGIN_DEG:
            return None, "解超关节限位或奇异，扫臂角也没有合格的"
        return None, f"{self.check_q(r['q'], floor_z)}，扫臂角也避不开"

    # ---- 路径
    def joint_path(self, qs, n=40, floor_z=None):
        """依次经过 qs 的关节直线插值（setPln_joint 的走法）途中有没有碰撞。None = 没有。"""
        for k in range(len(qs) - 1):
            q0, q1 = np.asarray(qs[k], float), np.asarray(qs[k + 1], float)
            for s in np.linspace(0.0, 1.0, n):
                why = self.check_q(q0 + s * (q1 - q0), floor_z)
                if why:
                    return f"第 {k + 1} 段关节插值 {s:.0%} 处{why}"
        return None

    def line(self, q0, T_end, vel=LINE_VEL, acc=LINE_ACC, floor_z=None, keep=False):
        """SDK movLA：q0 处的 TCP → T_end（世界，米）走直线。返回 (关节点列 (N,7) 度, pset, 原因)。
        原因不是 None 或 keep=False 时 pset 当场销毁、返回 None；keep=True 时调用方用完必须 destroy(pset)。"""
        q0 = np.asarray(q0, float)
        start = T_to_xyzabc(self.T_bw @ _m2mm(self.tcp(q0)))
        end = T_to_xyzabc(self.T_bw @ _m2mm(T_end))
        kk = self.k.k
        kk.set_tool_kine(xyzabc_to_T(TOOL).tolist())
        try:
            with quiet():
                pts, pset = kk.movLA([float(v) for v in start], [float(v) for v in end], q0.tolist(),
                                     float(vel), float(acc), 50)
        finally:
            kk.remove_tool_kine()
        if pset is None or not len(pts):
            return None, None, "SDK movLA 规划失败（直线途中超限、奇异或够不着）"
        P = np.asarray(pts, float)
        why = None
        if np.any(P > self.lim[:, 0] - 0.5) or np.any(P < self.lim[:, 1] + 0.5):
            why = "直线途中关节贴到软限位"
        if why is None and len(P) > 1 and np.abs(np.diff(P, axis=0)).max() > 2.0:
            why = f"直线途中关节一步跳 {np.abs(np.diff(P, axis=0)).max():.1f}°（过奇异附近）"
        if why is None:
            for q in list(P[:: max(1, len(P) // 20)]) + [P[-1]]:
                w = self.check_q(q, floor_z)
                if w:
                    why = f"直线途中{w}"
                    break
        if why is None:
            e = np.linalg.norm(self.tcp(P[-1])[:3, 3] - np.asarray(T_end, float)[:3, 3])
            if e > 0.002:
                why = f"直线终点和目标差 {e * 1000:.1f}mm"
        if why is not None or not keep:
            kk.destroy_point_set(pset)
            pset = None
        return P, pset, why

    def destroy(self, pset):
        if pset is not None:
            self.k.k.destroy_point_set(pset)

    # ---- 抓取
    @staticmethod
    def grasp_T(c, R, support_z):
        """抓取点 c（世界，米：物体上要夹的那一点，两指之间）+ 姿态 R → TCP 位姿。
        斜着抓时指尖比 TCP 低 TIP·cosθ，所以 TCP 沿接近方向往回退 d，让指尖离支撑面 TIP_CLEAR；
        c 落在 TCP 前方 d 处、离指尖 TIP − d，必须还在能夹的那段（≤ PAD）里。返回 (T, d, 原因)。"""
        c = np.asarray(c, float)
        a = R[:, 2]
        cz = -a[2]                                     # cos θ
        if cz > 1e-3:
            d = TIP_M - (c[2] - support_z - TIP_CLEAR_M) / cz
        else:
            d = -np.inf if c[2] - support_z >= TIP_CLEAR_M else np.inf
        d = max(d, TIP_M - PAD_M)
        if d > TIP_M - 0.003:
            th = np.degrees(np.arccos(np.clip(cz, -1, 1)))
            return None, None, f"太矮：斜 {th:.0f}° 时指尖要离支撑面 {TIP_CLEAR_M * 1000:.0f}mm，夹持点落到指尖外面"
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, c - d * a
        return T, float(d), None

    def plan_grasp(self, c, support_z, width_at, full_open=FULL_OPEN_M, obstacle_top=None, tilts=TILTS):
        """c：抓取点（世界，米）；support_z：物体下面支撑面的世界 z；width_at(α 度) → 物体沿水平方向 α 的宽度（米）或 None；
        obstacle_top：335 看到的接近走廊里最高点的世界 z（None = 不知道）。
        开合方向按「宽度从窄到宽」试（细长物体先试横着夹），每个方向先 45° 后 60°，倾斜方向先试顺着肩往外的。
        成功返回 plan dict；全部不行 raise RuntimeError（附各条原因的计数）。"""
        c = np.asarray(c, float)
        out_az = float(np.degrees(np.arctan2(c[1] - SHOULDER_W[1] / 1000, c[0] - SHOULDER_W[0] / 1000)))
        alphas = []
        why = Counter()
        for al in range(0, 180, 15):
            w = width_at(al)
            if w is not None and w > full_open - PRE_GAP_M:
                why[f"沿 {al}° 宽 {w * 1000:.0f}mm 放不进（满开 {full_open * 1000:.0f} − 预留 {PRE_GAP_M * 1000:.0f}）"] += 1
                continue
            alphas.append((np.inf if w is None else w, al))
        alphas.sort()
        for w, al in alphas:
            for th in tilts:
                for lean, psi in candidates(th, al, out_az):
                    R = grasp_R(th, lean, psi)
                    T, d, y = self.grasp_T(c, R, support_z)
                    if y:
                        why[y] += 1
                        continue
                    q_g, y = self.solve(T, READY_Q, support_z)
                    if q_g is None:
                        why["抓取位" + y] += 1
                        continue
                    p = self._approach(T, q_g, support_z, obstacle_top, why)
                    if p is None:
                        continue
                    p.update(T_grasp=T, q_grasp=q_g, theta=th, alpha=al, lean_az=lean, psi=psi, d=d,
                             width=None if not np.isfinite(w) else float(w), c=c, support_z=support_z)
                    return p
        top = "；".join(f"{k}（{n}）" for k, n in why.most_common(4))
        raise RuntimeError(f"试过 {len(alphas)} 个开合方向 × 倾角 {list(tilts)}，都不行：{top}")

    def _approach(self, T, q_g, support_z, obstacle_top, why):
        """抓取位 → 预抓取点（沿接近方向退）→ 上方点（竖直抬到过渡高度）→ 观察位的关节路径。"""
        a = T[:3, 2]
        for back in PRE_BACKS:
            T_pre = T.copy()
            T_pre[:3, 3] = T[:3, 3] - back * a
            q_p, y = self.solve(T_pre, q_g, support_z)
            if q_p is None:
                why[f"预抓取点（退 {back * 100:.0f}cm）{y}"] += 1
                continue
            _, _, y = self.line(q_p, T, vel=APPROACH_VEL, floor_z=support_z)
            if y:
                why[f"预抓取 → 抓取{y}"] += 1
                continue
            need = T_pre[2, 3] if obstacle_top is None else max(T_pre[2, 3], obstacle_top + CLEAR_M)
            zs = [max(TRANSIT_Z_M, need)]
            zs += [z for z in np.arange(zs[0] - 0.02, need - 1e-9, -0.02)] + ([need] if need < zs[0] else [])
            for z in zs:
                T_ab = T_pre.copy()
                T_ab[2, 3] = z
                q_a, y = self.solve(T_ab, q_p)
                if q_a is None:
                    why[f"上方点 z={z * 1000:.0f}mm {y}"] += 1
                    continue
                if z > T_pre[2, 3] + 1e-4:
                    _, _, y = self.line(q_a, T_pre)
                    if y:
                        why[f"上方点 → 预抓取{y}"] += 1
                        continue
                via = None
                y = self.joint_path([HOME_Q, q_a])
                if y:
                    y2 = self.joint_path([HOME_Q, READY_Q, q_a])
                    if y2:
                        why[f"观察位 → 上方点{y}"] += 1
                        continue
                    via = READY_Q.copy()
                return dict(T_pre=T_pre, q_pre=q_p, back=back, T_above=T_ab, q_above=q_a, via=via,
                            note="" if z >= zs[0] - 1e-9 else f"（过渡高度 {zs[0] * 1000:.0f}mm 够不着，降到 {z * 1000:.0f}mm）")
        return None


def describe(p) -> str:
    return (f"倾 {p['theta']:.0f}°、开合方向 {p['alpha']}°（倾斜朝 {p['lean_az']:.0f}°）；"
            f"TCP {fmt_mm(p['T_grasp'][:3, 3])}mm（沿接近方向比抓取点退 {p['d'] * 1000:.0f}mm，指尖离支撑面 ≥{TIP_CLEAR_M * 1000:.0f}mm）；"
            f"预抓取退 {p['back'] * 100:.0f}cm；上方点 z={p['T_above'][2, 3] * 1000:.0f}mm{p['note']}"
            + ("；经 READY 中转" if p["via"] is not None else ""))


# ================================================================ 335 外参（世界系）

def load_extrinsic():
    """calibrate_top335_marvin 的结果：335 系点 = R·世界点 + t（米）。
    和旧 top335 的形式一样（旧的是「R·法兰 + t」），所以 top335 里 to_flange / measure_mask / locate_click /
    scene_obstacle_z / support_flange_z 原样能用，只是「法兰坐标」现在就是世界系 TCP 坐标。"""
    if not EXTRINSIC.exists():
        raise FileNotFoundError(f"没有 {EXTRINSIC.name}：先跑 python -m robojev.calibrate_top335_marvin")
    d = json.loads(EXTRINSIC.read_text())
    return np.asarray(d["R"], float), np.asarray(d["t"], float), d


# ================================================================ 硬件：臂

class MarvinArm:
    """B 臂。with MarvinArm(planner) as arm: ...  退出时停规划、下伺服（抱闸，和 天机Marvin 的工具一样）、release。"""

    def __init__(self, planner: Planner, ip: str = DEFAULT_IP, sdk: str = str(DEFAULT_SDK)):
        self.p, self.ip, self.sdk = planner, ip, sdk
        self.m = None

    def __enter__(self):
        self.m = Marvin(self.sdk, self.ip).__enter__()      # 连不上 / 数据帧不刷新会 SystemExit
        try:
            self._check_and_enable()
        except BaseException:
            self.m.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        if self.m is None:
            return
        try:
            self.m.robot.stopRunPln_joint(ARM)
            self.m.disable(ARM)
        finally:
            self.m.__exit__(None, None, None)
            self.m = None

    def _frame(self):
        d = self.m.sub()
        s, o = d["states"][1], d["outputs"][1]
        return s["cur_state"], s["err_code"], np.array(o["fb_joint_pos"], float), o["low_speed_flag"], o["traj_state"], d

    def _check_and_enable(self):
        st, err, q, _, _, d = self._frame()
        if err != 0 or st == 100:
            raise RuntimeError(f"B 臂有错误 state={st} err={err}：不自动清（err 13 = 急停按着）。处理好再跑")
        tool = np.array(d["inputs"][1]["tool_kine"], float)
        if np.abs(tool - TOOL).max() > 0.5:
            raise RuntimeError(f"控制器里 B 臂工具是 {np.round(tool, 1).tolist()}，不是夹持中心 {TOOL}："
                               f"控制器重启过就跑 天机Marvin/marvin_tool.py apply")
        if not self.m.robot.pln_init(str(Path(self.sdk) / KINE_CFG)):
            raise RuntimeError("SDK pln_init 失败（关节规划要它）")
        if st != 1:
            if not self.m.enable_position(ARM, vel=POS_MODE_PCT, acc=POS_MODE_PCT):
                raise RuntimeError("B 臂进不了位置模式")
        else:
            self.m.batch((self.m.robot.set_vel_acc, (ARM, POS_MODE_PCT, POS_MODE_PCT)))
        print(f"B 臂已连接：位置模式，关节 {np.round(q, 1).tolist()}，TCP 世界 {fmt_mm(self.tcp()[:3, 3])}mm")

    def q(self) -> np.ndarray:
        return self._frame()[2]

    def tcp(self) -> np.ndarray:
        return self.p.tcp(self.q())

    def stop(self):
        try:
            self.m.robot.stopRunPln_joint(ARM)
        finally:
            self.m.robot.soft_stop(ARM)

    def _wait(self, q_end, lo, hi, label, timeout):
        t0, started = time.time(), False
        while time.time() - t0 < timeout:
            st, err, q, low, traj, _ = self._frame()
            if st != 1 or err != 0:
                self.stop()
                raise RuntimeError(f"[{label}] B 臂状态异常 state={st} err={err}，已停")
            if np.any(q < lo) or np.any(q > hi):
                self.stop()
                raise RuntimeError(f"[{label}] 关节跑出规划包络 ±{GUARD_DEG}°，已软急停 q={np.round(q, 2).tolist()}")
            if traj != b"\x00":
                started = True
            if (started or time.time() - t0 > 1.0) and traj == b"\x00" and low == b"\x01" \
                    and np.abs(q - q_end).max() < 0.2:
                return time.time() - t0
            time.sleep(0.01)
        self.stop()
        raise RuntimeError(f"[{label}] {timeout:.0f}s 没到位，已停 q={np.round(self.q(), 2).tolist()}")

    def move_joints(self, q_target, label, vel=JOINT_VEL, acc=JOINT_ACC, check=True):
        """SDK setPln_joint（关节空间规划，位置模式）。动之前查一遍关节插值路径。"""
        q0, qt = self.q(), np.asarray(q_target, float)
        if np.abs(qt - q0).max() < 0.2:
            return 0.0
        if check:
            why = self.p.joint_path([q0, qt])
            if why:
                raise RuntimeError(f"[{label}] 不走：{why}")
        span = np.abs(qt - q0).max()
        timeout = 10.0 + 3.0 * span / (180.0 * vel)
        if not self.m.robot.setPln_joint(ARM, q0.tolist(), qt.tolist(), float(vel), float(acc)):
            raise RuntimeError(f"[{label}] setPln_joint 没收下")
        el = self._wait(qt, np.minimum(q0, qt) - GUARD_DEG, np.maximum(q0, qt) + GUARD_DEG, label, timeout)
        print(f"   [{label}] 到位 {el:.1f}s")
        return el

    def move_line(self, T_end, label, vel=LINE_VEL, floor_z=None):
        """SDK movLA → setPln_Cart：TCP 走直线到 T_end（世界，米）。规划不合格就不动。"""
        q0 = self.q()
        P, pset, why = self.p.line(q0, T_end, vel=vel, floor_z=floor_z, keep=True)
        if why:
            raise RuntimeError(f"[{label}] 不走：{why}")
        try:
            dist = np.linalg.norm(np.asarray(T_end)[:3, 3] - self.p.tcp(q0)[:3, 3])
            self.m.robot.setPln_Cart(ARM, pset)
            el = self._wait(P[-1], P.min(0) - GUARD_DEG, P.max(0) + GUARD_DEG, label, 10.0 + 3.0 * len(P) / 50.0)
        finally:
            self.p.destroy(pset)
        print(f"   [{label}] 直线 {dist * 1000:.0f}mm 到位 {el:.1f}s")
        return el

    def go_home(self, label="回观察位"):
        """当前 → 观察位（竖直向上）。直接关节插值碰撞就经 READY 中转。"""
        q0 = self.q()
        if self.p.joint_path([q0, HOME_Q]) is None:
            return self.move_joints(HOME_Q, label)
        if self.p.joint_path([q0, READY_Q, HOME_Q]) is None:
            self.move_joints(READY_Q, label + "（经 READY）")
            return self.move_joints(HOME_Q, label)
        raise RuntimeError(f"[{label}] 从当前姿态回观察位的关节路径都会碰：{self.p.joint_path([q0, HOME_Q])}")


# ================================================================ 硬件：夹爪

class Gripper:
    """HKVR TG-9801，PC 上的 Prolific RS485，Modbus 从站 1，**115200**（这台的实测值，不是手册默认 1M）。
    值的约定和旧流程一样：0 = 全开，1 = 合拢（HKVRHand.set_opening 的原始位置 / 1000）。"""

    def __init__(self, force: int = 50, speed: int = 1000):
        from atom_robot.ar5.config import HKVRConfig
        from atom_robot.ar5.drivers.hkvr_hand import HKVRHand

        cfg = HKVRConfig(baudrate=115200)
        cfg.force, cfg.speed = int(force), int(speed)
        self.hand = HKVRHand(cfg)

    def __enter__(self):
        self.hand.connect()
        return self

    def __exit__(self, *exc):
        self.hand.disconnect()

    def get(self) -> float:
        return float(self.hand.get_opening())

    def to(self, v: float, wait: float = 1.5, force=None, speed=None):
        if force is not None or speed is not None:
            self.hand.set_grip_params(force=force, speed=speed)
            time.sleep(0.15)                 # 串口线程在下一次下发前写入
        self.hand.set_opening(v)
        time.sleep(wait)
        return self.get()

    def for_width(self, width_m: float, full_open: float = FULL_OPEN_M) -> float:
        """张开到「两指间距 = width_m」对应的值。"""
        return float(np.clip(1.0 - width_m / full_open, 0.0, 1.0))
