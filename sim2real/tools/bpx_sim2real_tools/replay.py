"""离线策略回放；复用部署 Actor，不连接 SDK，也不执行起身。"""

from bpx_runtime.jsonl import JsonlWriter
from bpx_sim2real.config import vector
from bpx_sim2real.controller import ActorController
from bpx_sim2real.safety import StateGuard
from .recording import load_samples, resample


def replay(config, input_path, output_path, command, *, policies=None):
    metadata, samples = load_samples(input_path)
    first_complete = next((i for i, s in enumerate(samples)
                           if all(v is not None for v in (s.q, s.dq, s.quat_xyzw, s.omega))
                           and (not config.policy["include_base_lin_vel"] or s.velocity_body is not None)), None)
    if first_complete is None:
        raise ValueError("录制没有完整 Actor 状态")
    samples = samples[first_complete:]
    guard = StateGuard(config)
    controller = ActorController(config, policies=policies)
    count = 0
    with JsonlWriter(output_path, {"mode": "replay", "input": str(input_path), "input_metadata": metadata,
                                 "config": str(config.path), "skipped_initial_samples": first_complete,
                                 "feedback": "recorded trajectory; policy targets do not alter recorded states"}) as log:
        for sample in resample(samples, config.period):
            guard.check(sample)
            result = controller.step(sample, command)
            target = vector(result["target_model"], 12, "target")
            controller.commit_target(target)
            result.update(target_model=target.tolist(), target_sdk=controller.calibration.position_to_sdk(target).tolist(),
                          host_time=sample.host_time, sample=sample.to_dict())
            log.write({"type": "policy", **result})
            count += 1
    return count
