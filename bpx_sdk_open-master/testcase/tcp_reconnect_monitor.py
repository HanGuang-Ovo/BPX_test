import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "example"))

import bpx_sdk
from example_options import parse_options


def main():
    options = parse_options(False)
    robot_state = bpx_sdk.RequestRobotState()
    robot_state.setRobotIp(options.robot_ip)
    robot_state.setRobotStateUploadPort(options.robot_state_port)
    robot_state.setTcpLocalPort(options.tcp_local_port)
    robot_state.setRobotStateUploadRate(options.state_rate_hz)

    print(f"bpx_sdk version={getattr(bpx_sdk, '__version__', 'unknown')}")
    print("TCP reconnect monitor: no motion or joint command will be sent.")
    print(
        f"robot_ip={options.robot_ip} "
        f"state_port={options.robot_state_port} "
        f"tcp_local_port={options.tcp_local_port} "
        f"state_rate={options.state_rate_hz}"
    )

    if not robot_state.connect():
        raise RuntimeError("failed to start request robot state connection")

    started_at = time.monotonic()
    was_connected = robot_state.isConnected()
    has_connected_once = was_connected
    disconnected_at = started_at
    print(f"[0 ms] TCP {'connected' if was_connected else 'connecting'}")

    try:
        while True:
            time.sleep(0.1)
            connected = robot_state.isConnected()
            if connected == was_connected:
                continue

            now = time.monotonic()
            elapsed_ms = round((now - started_at) * 1000)
            if not connected:
                disconnected_at = now
                print(f"[{elapsed_ms} ms] TCP disconnected; retrying in background")
            elif has_connected_once:
                downtime_ms = round((now - disconnected_at) * 1000)
                print(f"[{elapsed_ms} ms] TCP reconnected; downtime={downtime_ms} ms")
            else:
                print(f"[{elapsed_ms} ms] TCP connected")
                has_connected_once = True
            was_connected = connected
    finally:
        robot_state.disconnect()


if __name__ == "__main__":
    main()
