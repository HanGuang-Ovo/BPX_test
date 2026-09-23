"""TorchScript、ONNX Runtime 与测试策略后端。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


class Policy(Protocol):
    def __call__(self, observation: FloatArray) -> npt.NDArray[np.float32]: ...


class ZeroPolicy:
    """用于验证 MuJoCo 模型和 PD 控制器，不代表训练策略。"""

    def __init__(self, action_dimension: int):
        self.action_dimension = action_dimension

    def __call__(self, observation: FloatArray) -> npt.NDArray[np.float32]:
        del observation
        return np.zeros(self.action_dimension, dtype=np.float32)


class ConstantPolicy:
    """忽略观测并返回固定动作，用于在等待 Init 时保持复位关节姿态。"""

    def __init__(self, action: npt.ArrayLike):
        value = np.asarray(action, dtype=np.float32)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("固定策略动作必须是一维有限数组")
        self.action = value.copy()

    def __call__(self, observation: FloatArray) -> npt.NDArray[np.float32]:
        del observation
        return self.action.copy()


class TorchScriptPolicy:
    def __init__(self, path: Path, observation_dimension: int, action_dimension: int, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "TorchScript 后端需要 PyTorch。请在包含 torch 和 mujoco 的同一 Python 环境中运行。"
            ) from exc
        self.path = path.expanduser().resolve()
        self.torch = torch
        self.device = device
        self.observation_dimension = observation_dimension
        self.action_dimension = action_dimension
        self.module = torch.jit.load(str(path), map_location=device).eval()

    def __call__(self, observation: FloatArray) -> npt.NDArray[np.float32]:
        array = np.asarray(observation, dtype=np.float32).reshape(1, self.observation_dimension)
        tensor = self.torch.from_numpy(array).to(self.device)
        with self.torch.inference_mode():
            output = self.module(tensor)
        action = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return _validate_action(action, self.action_dimension)


class OnnxPolicy:
    def __init__(self, path: Path, observation_dimension: int, action_dimension: int):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX 后端需要 onnxruntime：python -m pip install onnxruntime") from exc
        self.path = path.expanduser().resolve()
        self.observation_dimension = observation_dimension
        self.action_dimension = action_dimension
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, observation: FloatArray) -> npt.NDArray[np.float32]:
        array = np.asarray(observation, dtype=np.float32).reshape(1, self.observation_dimension)
        output = self.session.run([self.output_name], {self.input_name: array})[0]
        action = np.asarray(output, dtype=np.float32).reshape(-1)
        return _validate_action(action, self.action_dimension)


class SingleFrameOnnxPolicy(OnnxPolicy):
    """将完整单帧观测适配到 48 维旧专家或 45 维无基座线速度专家。"""

    def __init__(self, path: Path, observation_dimension: int, action_dimension: int):
        super().__init__(path, observation_dimension, action_dimension)
        self.full_observation_dimension = observation_dimension
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 2 or shape[1] not in (observation_dimension, observation_dimension - 3):
            raise ValueError(
                f"单帧专家输入应为 {observation_dimension} 或 {observation_dimension - 3} 维，实际为 {shape}"
            )
        self.observation_dimension = shape[1]

    def __call__(self, observation: FloatArray) -> npt.NDArray[np.float32]:
        array = np.asarray(observation, dtype=np.float32).reshape(self.full_observation_dimension)
        if self.observation_dimension == self.full_observation_dimension - 3:
            array = array[3:]
        return super().__call__(array)


def _validate_action(action: npt.NDArray[np.float32], action_dimension: int) -> npt.NDArray[np.float32]:
    if action.shape != (action_dimension,):
        raise RuntimeError(f"策略输出维度不匹配：期望 {action_dimension}，实际 {action.shape}")
    if not np.all(np.isfinite(action)):
        raise FloatingPointError("策略动作中出现 NaN 或 Inf")
    return action


def create_policy(
    backend: str,
    torchscript_path: Path,
    onnx_path: Path,
    observation_dimension: int,
    action_dimension: int,
    device: str = "cpu",
) -> Policy:
    if backend == "zero":
        return ZeroPolicy(action_dimension)
    if backend == "torchscript":
        return TorchScriptPolicy(torchscript_path, observation_dimension, action_dimension, device)
    if backend == "onnx":
        return OnnxPolicy(onnx_path, observation_dimension, action_dimension)
    raise ValueError(f"不支持的策略后端：{backend}")
