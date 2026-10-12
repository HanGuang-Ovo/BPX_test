"""兼容原有 sim2sim 手柄入口；通用实现由仿真与实机窗口共用。"""

from bpx_runtime.gamepad import (
    ABS_INFO_STRUCT,
    ABS_RX,
    ABS_X,
    ABS_Y,
    BTN_TL,
    BTN_TR,
    EVENT_STRUCT,
    EV_ABS,
    EV_KEY,
    KEY_STATE_BYTES,
    AxisCalibration,
    GamepadCommandSource,
    GamepadMapping,
    InputDevice,
    enumerate_gamepads,
    select_gamepad,
)

__all__ = [
    "ABS_INFO_STRUCT",
    "ABS_RX",
    "ABS_X",
    "ABS_Y",
    "BTN_TL",
    "BTN_TR",
    "EVENT_STRUCT",
    "EV_ABS",
    "EV_KEY",
    "KEY_STATE_BYTES",
    "AxisCalibration",
    "GamepadCommandSource",
    "GamepadMapping",
    "InputDevice",
    "enumerate_gamepads",
    "select_gamepad",
]
