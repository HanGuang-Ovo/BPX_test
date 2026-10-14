"""实机执行循环：仅验证有限数与四元数，不限幅、限速或限矩。"""
"""管理控制循环和时间。 等待完整反馈，周期性读状态、更新控制器、发送目标并记录日志。"""
"""目前默认策略更新 50 Hz、发送 100 Hz。"""
import time

import numpy as np

from bpx_runtime.jsonl import JsonlWriter
from .config import vector
from .behavior import HardwareController
from .safety import StateGuard
from .sdk_robot import SDKSource


def wait_for_feedback(source, guard, timeout, *, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    while clock() < deadline:
        sample = source.read()
        needed = [sample.q, sample.dq, sample.quat_xyzw, sample.omega]
        if guard.config.policy["include_base_lin_vel"]:
            needed.append(sample.velocity_body)
        if all(v is not None for v in needed):
            guard.check(sample)
            return sample
        sleep(.005)
    raise TimeoutError("等待首个完整 Actor 反馈超时")


def run_live(config, output_path, command, duration, *, policies=None, source_factory=SDKSource,
            clock=time.monotonic, sleep=time.sleep, command_source=None, init_request_source=None,
            status_callback=None):
    if duration is not None and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("duration 必须为有限正数")
    command = vector(command, 3, "command")
    controller = HardwareController(config, policies=policies)
    guard = StateGuard(config)
    cal = controller.calibration
    zeros = np.zeros(12)
    send_period = 1. / config.control["send_rate_hz"]
    with JsonlWriter(output_path, {"mode": "live", "config": str(config.path), "runtime": config.runtime,
                                "calibration": config.calibration, "duration_s": duration,
                                "command_source": "dynamic" if command_source is not None else "constant"}) as log:
        with source_factory(config, joint_control=True) as source:
            sample = wait_for_feedback(source, guard, config.network["connect_timeout_s"], clock=clock, sleep=sleep)
            controller.initialize(sample)
            start = next_send = next_policy = clock()
            policy_count = 0
            previous_mode = None
            while True:
                if next_send > clock():
                    sleep(next_send - clock())
                now = clock()
                if duration is not None and now - start + 1e-9 >= duration:
                    break
                sample = source.read()
                guard.check(sample)
                policy_due = now + 1e-9 >= next_policy
                init_requested = False
                if policy_due:
                    if command_source is not None:
                        command = vector(command_source(), 3, "command")
                    init_requested = bool(init_request_source()) if init_request_source is not None else False
                    policy_count += 1
                    next_policy += config.period
                result = controller.step(sample, command, now=now, policy_due=policy_due,
                                        init_requested=init_requested)
                target = vector(result["target_model"], 12, "target")
                source.send(cal.gains_to_sdk(controller.kp), cal.position_to_sdk(target),
                            cal.gains_to_sdk(controller.kd), zeros, zeros)
                controller.commit_target(target)
                after = clock()
                if controller.mode != previous_mode:
                    if status_callback is not None:
                        status_callback(controller.mode, result["reason"])
                    previous_mode = controller.mode
                log.write({"type": "command", "host_time": after, "mode": controller.mode,
                            "raw_command": np.asarray(command).tolist(), "init_requested": init_requested,
                            "target_model": target.tolist(), "target_sdk": cal.position_to_sdk(target).tolist(),
                            "kp_model": controller.kp.tolist(), "kd_model": controller.kd.tolist(),
                            "sample": sample.to_dict(), "policy": result})
                next_send += send_period
                if after > next_send:
                    next_send = after + send_period
                if after > next_policy:
                    next_policy = after + config.period
    return policy_count
