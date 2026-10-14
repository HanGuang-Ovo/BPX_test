"""JSONL 录制读取、离线重采样和诊断报告。"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from bpx_sim2real.state import Sample


def load_samples(path):
    metadata = {}
    samples = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("type") == "metadata":
                if row.get("schema") != 1:
                    raise ValueError("不支持的录制 schema")
                metadata = row
            elif row.get("type") == "sample":
                samples.append(Sample.from_dict(row["sample"]))
    if not samples:
        raise ValueError("录制中没有 sample")
    times = np.array([s.host_time for s in samples])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("录制主机时间必须为严格递增有限值")
    return metadata, samples


def resample(samples, period):
    """策略时刻选最新的过去帧，禁止读取未来状态；保持帧同时增加年龄。"""
    index = 0
    start, end = samples[0].host_time, samples[-1].host_time
    ticks = int(np.floor((end - start) / period + 1e-8)) + 1
    for tick in range(ticks):
        now = start + tick * period
        while index + 1 < len(samples) and samples[index + 1].host_time <= now + 1e-9:
            index += 1
        sample = samples[index]
        hold = max(0.0, now - sample.host_time)
        ages = {
            name: None if getattr(sample, name) is None else getattr(sample, name) + hold
            for name in ("joint_age_s", "imu_age_s", "odom_age_s")
        }
        telemetry = dict(sample.telemetry)
        if telemetry.get("motion_age_s") is not None:
            telemetry["motion_age_s"] += hold
        yield replace(sample, host_time=now, telemetry=telemetry, **ages)


def summarize(samples):
    duration = samples[-1].host_time - samples[0].host_time
    report = {
        "samples": len(samples),
        "duration_s": duration,
        "disconnected_samples": sum(not s.connected for s in samples),
        "groups": {},
    }
    for group in ("joint", "imu", "odom"):
        previous = None
        changes = 0
        ages = []
        for s in samples:
            token = getattr(s, f"{group}_token")
            key = (s.session, token)
            if token is not None and key != previous:
                changes += 1
                previous = key
            age = getattr(s, f"{group}_age_s")
            if age is not None:
                ages.append(age)
        report["groups"][group] = {
            "observed_updates": changes,
            "observed_hz": max(0, changes - 1) / duration if duration > 0 else None,
            "max_age_s": max(ages) if ages else None,
        }
    q = np.array([s.q for s in samples if s.q is not None], dtype=float)
    if len(q):
        report["sdk_q_mean"] = q.mean(axis=0).tolist()
        report["sdk_q_std"] = q.std(axis=0).tolist()
        report["sdk_q_min"] = q.min(axis=0).tolist()
        report["sdk_q_max"] = q.max(axis=0).tolist()
    report["identities"] = list({json.dumps(s.identity, sort_keys=True) for s in samples if s.identity})
    return report
