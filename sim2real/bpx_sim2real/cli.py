"""分阶段命令行。check/analyze/calibrate/replay 不实例化 SDK。"""

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from bpx_runtime.joints import SDK_JOINT_NAMES, POLICY_JOINT_NAMES
from .calibration import Calibration
from .config import load_config, positive
from .recording import JsonlWriter, load_samples, summarize
from .runner import jog, replay, run_live
from .sdk_robot import SDKSource


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/runtime.toml"
if not DEFAULT_CONFIG.is_file():
    DEFAULT_CONFIG = Path(sys.prefix) / "share/bpx-deployment/config/runtime.toml"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def check(config):
    models = {}
    for mode in ("walk", "stand"):
        p = config.model_path(mode)
        models[mode] = {"path": str(p), "exists": p.is_file(),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("numpy", "onnxruntime", "bpx_sdk")}
    return {"config": str(config.path), "calibration": str(config.calibration_path),
            "dependencies": dependencies, "models": models,
            "observe_blockers": [] if dependencies["bpx_sdk"] else ["当前 Python 未安装 bpx_sdk"],
            "replay_blockers": [] if dependencies["onnxruntime"] else ["当前 Python 未安装 onnxruntime"],
            "deploy_blockers": config.blockers("deploy"),
            "mapping": [{"policy_index": i, "name": j.name, "sdk_index": j.sdk_index,
                         "sdk_documented_name": SDK_JOINT_NAMES[j.sdk_index],
                         "sign": j.sign, "offset": j.offset, "confirmed": all((j.mapping_verified,
                         j.direction_verified, j.offset_verified, j.limits_verified))}
                        for i, j in enumerate(config.joints)]}


def capture(config, output, duration, poll_hz, *, source_factory=SDKSource):
    duration = positive(duration, "duration")
    period = 1. / positive(poll_hz, "poll_hz")
    samples = []
    with JsonlWriter(output, {"mode": "observe", "config": str(config.path),
                             "runtime": config.runtime, "calibration": config.calibration,
                             "created_at": datetime.now().astimezone().isoformat(),
                             "note": "RequestRobotState; SDK connect may perform automatic time synchronization"}) as log:
        with source_factory(config, joint_control=False) as source:
            start = time.monotonic()
            next_print = start
            tick = 0
            while time.monotonic()-start < duration:
                sample = source.read()
                samples.append(sample)
                log.write({"type": "sample", "sample": sample.to_dict()})
                now = time.monotonic()
                if now >= next_print:
                    print(json.dumps({"elapsed_s": round(now-start, 3), "connected": sample.connected,
                                      "identity": sample.identity, "sdk_q": sample.q,
                                      "joint_age_s": sample.joint_age_s, "imu_age_s": sample.imu_age_s}, ensure_ascii=False))
                    next_print = now+1.
                tick += 1
                time.sleep(max(0., start+tick*period-time.monotonic()))
    return summarize(samples)


def calibration_evidence(before_path, after_path, joint_name, model_delta, reference_angle=None):
    """两个人工摆出的静止姿态用于建议索引/符号；不会修改 TOML。"""
    if joint_name not in POLICY_JOINT_NAMES or not np.isfinite(model_delta) or abs(model_delta) < 1e-6:
        raise ValueError("必须指定关节名称及已知的非零模型角变化")
    _, before = load_samples(before_path)
    _, after = load_samples(after_path)
    a = np.array([s.q for s in before if s.connected and s.q is not None], dtype=float)
    b = np.array([s.q for s in after if s.connected and s.q is not None], dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("两个姿态各需要至少两个有效关节样本")
    identities = [s.identity for s in before+after if s.identity]
    serials = {identity.get("serial_number") for identity in identities}
    if len(serials) > 1:
        raise ValueError("两个录制来自不同机器人")
    delta = b.mean(axis=0)-a.mean(axis=0)
    index = int(np.argmax(np.abs(delta)))
    if abs(delta[index]) < 1e-6:
        raise ValueError("没有可检测关节角变化")
    sign = 1 if model_delta*delta[index] > 0 else -1
    others = np.delete(np.abs(delta), index)
    warnings = []
    if max(a.std(axis=0).max(), b.std(axis=0).max()) > .01:
        warnings.append("录制姿态不够静止；请只截取摆动前后的稳定区间")
    if others.max() > abs(delta[index])*.25:
        warnings.append("多个关节同时变化，不能据此唯一确认通道")
    if abs(abs(delta[index])-abs(model_delta)) > max(.02, abs(model_delta)*.2):
        warnings.append("实测角变化与参考角变化差异较大，需核查单位和参考姿态")
    if not serials or None in serials or "" in serials:
        warnings.append("录制缺少有效 SN，无法绑定实机身份")
    offset = None
    if reference_angle is not None:
        if not np.isfinite(reference_angle):
            raise ValueError("reference_angle 必须有限")
        offset = float(reference_angle-sign*a.mean(axis=0)[index])
    return {"joint": joint_name, "suggested_sdk_index": index, "suggested_sign": sign,
            "suggested_offset": offset, "sdk_delta": delta.tolist(), "model_delta": model_delta,
            "before_sdk_q": a.mean(axis=0).tolist(), "after_sdk_q": b.mean(axis=0).tolist(),
            "warnings": warnings, "verified": False,
            "note": "建议值需人工与物理姿态复核；工作限位需独立确定。TOML 未被修改。"}


def parser():
    p = argparse.ArgumentParser(description="BPX 分阶段 sim2real 工具")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="离线检查配置、模型和未确认项；不连接机器人")
    observe = sub.add_parser("inspect", help="只读状态录制；不发送运动命令")
    observe.add_argument("--output", type=Path, required=True)
    observe.add_argument("--duration", type=float, default=10.)
    observe.add_argument("--poll-hz", type=float, default=100.)
    analyze = sub.add_parser("analyze", help="分析只读录制")
    analyze.add_argument("recording", type=Path)
    analyze.add_argument("--output", type=Path)
    compare = sub.add_parser("compare-startup", help="比较多次上电的静止状态录制")
    compare.add_argument("recordings", nargs="+", type=Path)
    compare.add_argument("--output", type=Path)
    cal = sub.add_parser("calibrate", help="人工摆动前后两份录制，生成通道/方向/偏移建议")
    cal.add_argument("--before", type=Path, required=True)
    cal.add_argument("--after", type=Path, required=True)
    cal.add_argument("--joint", choices=POLICY_JOINT_NAMES, required=True)
    cal.add_argument("--model-delta", type=float, required=True, help="已知参考姿态的模型角变化（rad）")
    cal.add_argument("--reference-angle", type=float, help="before 姿态中该关节已知模型角（rad）")
    cal.add_argument("--output", type=Path, required=True)
    motion = sub.add_parser("jog", help="默认只生成点动计划；--live --arm 才发送命令")
    motion.add_argument("--joint", choices=POLICY_JOINT_NAMES, required=True)
    motion.add_argument("--delta", type=float, required=True, help="模型坐标系角变化（rad）")
    motion.add_argument("--live", action="store_true")
    motion.add_argument("--arm", action="store_true")
    motion.add_argument("--output", type=Path)
    off = sub.add_parser("replay", help="离线策略推理；不连接 SDK")
    off.add_argument("recording", type=Path)
    off.add_argument("--output", type=Path, required=True)
    off.add_argument("--enforce-limits", action="store_true", help="使用已填写的工作限位和速度约束")
    off.add_argument("--vx", type=float, default=0.)
    off.add_argument("--vy", type=float, default=0.)
    off.add_argument("--wz", type=float, default=0.)
    run = sub.add_parser("run", help="默认仅检查就绪项；实机需要 --live --arm")
    run.add_argument("--live", action="store_true")
    run.add_argument("--arm", action="store_true")
    run.add_argument("--duration", type=float, default=10.)
    run.add_argument("--output", type=Path)
    run.add_argument("--vx", type=float, default=0.)
    run.add_argument("--vy", type=float, default=0.)
    run.add_argument("--wz", type=float, default=0.)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "check":
            result = check(config)
        elif args.command == "inspect":
            result = capture(config, args.output, args.duration, args.poll_hz)
        elif args.command == "analyze":
            _, samples = load_samples(args.recording)
            result = summarize(samples)
            if args.output:
                write_json(args.output, result)
        elif args.command == "compare-startup":
            result = {"runs": [{"path": str(path), **summarize(load_samples(path)[1])} for path in args.recordings]}
            means = [r["sdk_q_mean"] for r in result["runs"] if "sdk_q_mean" in r]
            if len(means) >= 2:
                result["sdk_startup_range_rad"] = np.ptp(np.asarray(means), axis=0).tolist()
            if args.output:
                write_json(args.output, result)
        elif args.command == "calibrate":
            result = calibration_evidence(args.before, args.after, args.joint, args.model_delta, args.reference_angle)
            write_json(args.output, result)
        elif args.command == "jog":
            if args.live and args.output is None:
                raise ValueError("live 点动必须指定 --output 日志")
            result = jog(config, args.joint, args.delta, args.output, live=args.live, armed=args.arm)
        elif args.command == "replay":
            count = replay(config, args.recording, args.output, (args.vx, args.vy, args.wz), enforce_limits=args.enforce_limits)
            result = {"policy_steps": count, "output": str(args.output), "mode": "offline"}
        elif args.command == "run":
            if not args.live:
                result = {"mode": "preflight", "blockers": config.blockers("deploy"),
                          "note": "未创建 SDK 对象；--live --arm 后才进入关节控制"}
            else:
                if args.output is None:
                    raise ValueError("live 策略必须指定 --output 日志")
                count = run_live(config, args.output, (args.vx, args.vy, args.wz), positive(args.duration, "duration"), armed=args.arm)
                result = {"policy_steps": count, "output": str(args.output)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("操作已中断，SDK 上下文执行退出处理。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
