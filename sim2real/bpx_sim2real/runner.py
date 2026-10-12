"""真实时钟策略运行与离线回放；所有 live 条件在创建 SDK 对象前检查。"""

import time

import numpy as np

from .controller import ActorController
from .recording import JsonlWriter, load_samples, resample
from .safety import StateGuard, TargetGuard, require_ready
from .sdk_robot import SDKSource


def replay(config, input_path, output_path, command, *, policies=None, enforce_limits=False):
    metadata, samples = load_samples(input_path)
    # 初次连接 TCP 后 UDP 的各组可能尚未到齐，跳过不完整的启动前缀。
    first_complete = next((i for i, s in enumerate(samples)
                           if s.connected and all(v is not None for v in (
                               s.q, s.dq, s.quat_xyzw, s.omega, s.joint_age_s, s.imu_age_s))
                           and (not config.policy["include_base_lin_vel"]
                                or (s.velocity_body is not None and s.odom_age_s is not None))), None)
    if first_complete is None:
        raise ValueError("录制没有完整 Actor 状态")
    samples = samples[first_complete:]
    guard = StateGuard(config)
    controller = ActorController(config, policies=policies)
    target_guard = None
    count = 0
    with JsonlWriter(output_path, {"mode": "replay", "input": str(input_path), "input_metadata": metadata,
                                 "config": str(config.path), "calibration_confirmed": not config.blockers("deploy"),
                                 "skipped_initial_samples": first_complete,
                                 "feedback": "recorded trajectory; policy targets do not alter recorded states"}) as log:
        for sample in resample(samples, config.period):
            guard.check(sample)
            result = controller.step(sample, command)
            target = np.asarray(result["target_model"])
            if enforce_limits:
                if target_guard is None:
                    target_guard = TargetGuard(config, result["q_model"])
                target, tau, clipped = target_guard.preview(target, result["q_model"], result["dq_model"], config.period)
                target_guard.commit(target)
                result.update(estimated_torque=tau.tolist(), constrained=clipped)
            controller.commit_target(target)
            result.update(target_model=target.tolist(), target_sdk=controller.calibration.position_to_sdk(target).tolist(),
                          host_time=sample.host_time, sample=sample.to_dict())
            log.write({"type": "policy", **result})
            count += 1
    return count


def wait_for_feedback(source, guard, timeout, *, require_imu=True, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    last = None
    changes = 0
    while clock() < deadline:
        sample = source.read()
        if not sample.connected:
            raise RuntimeError("初始化期间断连")
        try:
            guard.check(sample, require_imu=require_imu, hardware=True)
        except (RuntimeError, ValueError):
            # 缺少启动反馈可以等待，已收到身份不匹配等确定错误立即拒绝。
            if sample.q is not None and (not require_imu or sample.quat_xyzw is not None):
                raise
            sleep(.005)
            continue
        if sample.joint_token != last:
            last = sample.joint_token
            changes += 1
        if changes >= 2:
            return sample
        sleep(.002)
    raise TimeoutError("等待至少两个新的高频反馈帧超时")


def run_live(config, output_path, command, duration, *, armed=False, policies=None, source_factory=SDKSource,
             clock=time.monotonic, sleep=time.sleep):
    require_ready(config, "deploy")
    if not armed:
        raise RuntimeError("实机策略需要显式 --live --arm，并从已确认站姿接管")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration 必须为有限正数")
    if np.asarray(command).shape != (3,) or not np.isfinite(command).all():
        raise ValueError("command 必须为三个有限数")
    controller = ActorController(config, policies=policies)  # 模型失败时尚未连接硬件。
    guard = StateGuard(config)
    cal = controller.calibration
    kp = cal.gains_to_sdk(config.control["kp"])
    kd = cal.gains_to_sdk(config.control["kd"])
    zeros = np.zeros(12)
    send_period = 1. / config.control["send_rate_hz"]
    max_late = min(config.safety["max_cycle_lateness_s"], config.period)
    with JsonlWriter(output_path, {"mode": "live", "config": str(config.path), "runtime": config.runtime,
                                 "calibration": config.calibration}) as log:
        with source_factory(config, joint_control=True) as source:
            sample = wait_for_feedback(source, guard, config.network["connect_timeout_s"], clock=clock, sleep=sleep)
            q = cal.position_from_sdk(sample.q)
            if np.any(np.abs(q-cal.default) > np.asarray(config.safety["takeover_position_error_rad"])):
                raise RuntimeError("实测姿态不符合已确认站立接管范围；本入口不自动起身")
            target_guard = TargetGuard(config, q)
            start = clock()
            send_index = policy_index = 0
            target = q.copy()
            result = None
            while True:
                deadline = start + send_index * send_period
                remaining = deadline-clock()
                if remaining > 0:
                    sleep(remaining)
                now = clock()
                if now-start + 1e-9 >= duration:
                    break
                if now-deadline > max_late:
                    raise RuntimeError("命令发送循环超期；停止当前控制")
                sample = source.read()
                guard.check(sample, clock(), hardware=True)
                if now + 1e-9 >= start + policy_index * config.period:
                    if now - (start + policy_index * config.period) > max_late:
                        raise RuntimeError("策略周期超期；禁止补跑过期历史帧")
                    result = controller.step(sample, command)
                    target = np.asarray(result["target_model"])
                    policy_index += 1
                q = cal.position_from_sdk(sample.q)
                dq = cal.velocity_from_sdk(sample.dq)
                processed, tau, constrained = target_guard.preview(target, q, dq, send_period)
                # 推理耗时计入期限与新鲜度，不能在超期后再发一条目标。
                after = clock()
                guard.check(sample, after, hardware=True)
                if after-deadline > max_late:
                    raise RuntimeError("推理/状态读取导致发送超期")
                source.send(kp, cal.position_to_sdk(processed), kd, zeros, zeros)
                target_guard.commit(processed)
                controller.commit_target(processed)
                log.write({"type": "command", "host_time": after, "mode": controller.mode,
                           "target_model": processed.tolist(), "target_sdk": cal.position_to_sdk(processed).tolist(),
                           "estimated_torque": tau.tolist(), "constrained": constrained,
                           "sample": sample.to_dict(), "policy": result if policy_index else None})
                send_index += 1
    return policy_index
