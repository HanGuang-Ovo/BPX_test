"""通过只读 SDK 采集反馈并保存录制。"""

from datetime import datetime
import json
import time

from bpx_runtime.jsonl import JsonlWriter
from bpx_sim2real.config import positive
from bpx_sim2real.sdk_robot import SDKSource
from .recording import summarize


def capture(config, output, duration, poll_hz, *, source_factory=SDKSource):
    duration = positive(duration, "duration")
    period = 1.0 / positive(poll_hz, "poll_hz")
    samples = []
    with JsonlWriter(
        output,
        {
            "mode": "observe",
            "config": str(config.path),
            "runtime": config.runtime,
            "calibration": config.calibration,
            "created_at": datetime.now().astimezone().isoformat(),
            "note": "RequestRobotState; SDK connect may perform automatic time synchronization",
        },
    ) as log:
        with source_factory(config, joint_control=False) as source:
            start = time.monotonic()
            next_print = start
            tick = 0
            while time.monotonic() - start < duration:
                sample = source.read()
                samples.append(sample)
                log.write({"type": "sample", "sample": sample.to_dict()})
                now = time.monotonic()
                if now >= next_print:
                    print(
                        json.dumps(
                            {
                                "elapsed_s": round(now - start, 3),
                                "connected": sample.connected,
                                "identity": sample.identity,
                                "sdk_q": sample.q,
                                "joint_age_s": sample.joint_age_s,
                                "imu_age_s": sample.imu_age_s,
                            },
                            ensure_ascii=False,
                        )
                    )
                    next_print = now + 1.0
                tick += 1
                time.sleep(max(0.0, start + tick * period - time.monotonic()))
    return summarize(samples)
