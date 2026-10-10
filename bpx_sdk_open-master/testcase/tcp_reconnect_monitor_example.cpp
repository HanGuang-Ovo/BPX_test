#include "bpx_sdk_version.h"
#include "example_options.h"
#include "request_robot_state.h"

#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
    bpx_sdk::example::Options options;
    if (!bpx_sdk::example::parseOptions(argc, argv, false, &options)) {
        return 1;
    }
    if (options.help_requested) {
        return 0;
    }

    bpx_sdk::RequestRobotState robot_state;
    robot_state.setRobotIp(options.robot_ip.c_str());
    robot_state.setRobotStateUploadPort(options.robot_state_port);
    robot_state.setTcpLocalPort(options.tcp_local_port);
    robot_state.setRobotStateUploadRate(options.state_rate_hz);

    std::cout << BPX_SDK_PROJECT_NAME << " version=" << BPX_SDK_PROJECT_VERSION << '\n'
              << "TCP reconnect monitor: no motion or joint command will be sent.\n"
              << "robot_ip=" << options.robot_ip
              << " state_port=" << options.robot_state_port
              << " tcp_local_port=" << options.tcp_local_port
              << " state_rate=" << options.state_rate_hz << '\n';

    if (!robot_state.connect()) {
        std::cerr << "failed to start request robot state connection" << std::endl;
        return 1;
    }

    const auto started_at = std::chrono::steady_clock::now();
    bool was_connected = robot_state.isConnected();
    bool has_connected_once = was_connected;
    auto disconnected_at = started_at;
    std::cout << "[0 ms] TCP " << (was_connected ? "connected" : "connecting") << std::endl;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const bool connected = robot_state.isConnected();
        if (connected == was_connected) {
            continue;
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - started_at).count();
        if (!connected) {
            disconnected_at = now;
            std::cout << '[' << elapsed_ms << " ms] TCP disconnected; retrying in background" << std::endl;
        } else if (has_connected_once) {
            const auto downtime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - disconnected_at).count();
            std::cout << '[' << elapsed_ms << " ms] TCP reconnected; downtime=" << downtime_ms << " ms" << std::endl;
        } else {
            std::cout << '[' << elapsed_ms << " ms] TCP connected" << std::endl;
            has_connected_once = true;
        }
        was_connected = connected;
    }
}
