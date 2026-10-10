#include "motion_level_control.h"
#include "motion_types.h"

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

namespace {

constexpr float kDefaultSpeedMps = 0.5f;
constexpr std::chrono::seconds kBoundPreparationDuration{2};
constexpr std::chrono::seconds kForwardPhaseDuration{2};
constexpr std::chrono::seconds kPausePhaseDuration{2};
constexpr std::chrono::seconds kStandUpTimeout{15};
constexpr std::chrono::milliseconds kPollInterval{200};
constexpr std::chrono::milliseconds kCommandFlushDelay{500};

volatile std::sig_atomic_t g_stop_requested = 0;

struct Options {
    std::string robot_ip;
    std::uint16_t state_port = bpx_sdk::DEFAULT_CLIENT_ROBOT_STATE_UDP_PORT;
    std::uint16_t tcp_local_port = 0;
    float speed_mps = kDefaultSpeedMps;
    bool help_requested = false;
};

void handleSignal(int) {
    g_stop_requested = 1;
}

void printUsage(const char* program) {
    std::cout
        << "Usage: " << program << " --robot-ip IP [options]\n\n"
        << "Real-robot options:\n"
        << "  --robot-ip IP          Robot IP address (required)\n"
        << "  --state-port PORT      Local robot-state UDP port (default: "
        << bpx_sdk::DEFAULT_CLIENT_ROBOT_STATE_UDP_PORT << ")\n"
        << "  --tcp-local-port PORT  Local TCP port, 0 for automatic selection\n"
        << "  --speed MPS            Forward speed, range 0.2..1.0 (default: "
        << kDefaultSpeedMps << ")\n"
        << "  -h, --help             Show this help\n";
}

bool requireValue(int argc, char** argv, int index) {
    if (index + 1 < argc) {
        return true;
    }
    std::cerr << argv[index] << " requires a value\n";
    return false;
}

bool parseUint16(const char* text, std::uint16_t* value) {
    if (!text || !value || text[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (*end != '\0' || parsed > std::numeric_limits<std::uint16_t>::max()) {
        return false;
    }
    *value = static_cast<std::uint16_t>(parsed);
    return true;
}

bool parseFloat(const char* text, float minimum, float maximum, float* value) {
    if (!text || !value || text[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    const float parsed = std::strtof(text, &end);
    if (*end != '\0' || !std::isfinite(parsed) ||
        parsed < minimum || parsed > maximum) {
        return false;
    }
    *value = parsed;
    return true;
}

bool parseOptions(int argc, char** argv, Options* options) {
    if (!options) {
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "-h" || argument == "--help") {
            printUsage(argv[0]);
            options->help_requested = true;
            return true;
        }
        if (argument == "--robot-ip") {
            if (!requireValue(argc, argv, i)) {
                return false;
            }
            options->robot_ip = argv[++i];
            continue;
        }
        if (argument == "--state-port") {
            if (!requireValue(argc, argv, i) ||
                !parseUint16(argv[++i], &options->state_port)) {
                std::cerr << "--state-port expects a port in range 0..65535\n";
                return false;
            }
            continue;
        }
        if (argument == "--tcp-local-port") {
            if (!requireValue(argc, argv, i) ||
                !parseUint16(argv[++i], &options->tcp_local_port)) {
                std::cerr << "--tcp-local-port expects a port in range 0..65535\n";
                return false;
            }
            continue;
        }
        if (argument == "--speed") {
            if (!requireValue(argc, argv, i) ||
                !parseFloat(argv[++i], 0.2f, 1.0f, &options->speed_mps)) {
                std::cerr << "--speed expects a value in range 0.2..1.0 m/s\n";
                return false;
            }
            continue;
        }
        std::cerr << "unknown option: " << argument << '\n';
        printUsage(argv[0]);
        return false;
    }

    if (options->robot_ip.empty()) {
        std::cerr << "--robot-ip is required for a real-robot test\n";
        printUsage(argv[0]);
        return false;
    }
    return true;
}

bool confirm(const std::string& prompt, const std::string& expected) {
    std::cout << prompt << "\n> " << std::flush;
    std::string input;
    return std::getline(std::cin, input) && input == expected;
}

void printState(const bpx_sdk::MotionLevelControl& motion) {
    std::uint8_t state = 0;
    std::uint8_t gait = 0;
    std::uint8_t sub_gait = 0;
    const bool has_state = motion.getCurrentMotionState(&state);
    const bool has_gait = motion.getCurrentGait(&gait);
    const bool has_sub_gait = motion.getSubGait(&sub_gait);

    std::cout << "robot feedback: state=";
    if (has_state) {
        std::cout << static_cast<unsigned int>(state);
    } else {
        std::cout << "unknown";
    }
    std::cout << " gait=";
    if (has_gait) {
        std::cout << static_cast<unsigned int>(gait);
    } else {
        std::cout << "unknown";
    }
    std::cout << " sub_gait=";
    if (has_sub_gait) {
        std::cout << static_cast<int>(static_cast<std::int8_t>(sub_gait));
    } else {
        std::cout << "unknown";
    }
    std::cout << '\n';
}

void disableVelocityControl(bpx_sdk::MotionLevelControl& motion) {
    std::cout << "Setting velocity control flag to false.\n";
    motion.setVelocityControlFlag(false);
    // Give the periodic command sender time to transmit the updated flag.
    std::this_thread::sleep_for(kCommandFlushDelay);
}

class SafeStopGuard {
public:
    explicit SafeStopGuard(bpx_sdk::MotionLevelControl& motion)
        : motion_(motion) {}

    ~SafeStopGuard() {
        motion_.setVelocityControlFlag(true);
        motion_.setVelocity(0.0f, 0.0f, 0.0f);
        std::this_thread::sleep_for(kCommandFlushDelay);
        motion_.setWalk();
        disableVelocityControl(motion_);
        std::cout << "Sending sit-down command before exit.\n";
        if (!motion_.setSitDown()) {
            std::cerr << "failed to send sit-down command during cleanup\n";
        }
        // Give the periodic command sender time to transmit sit-down before
        // the connection is closed.
        std::this_thread::sleep_for(kCommandFlushDelay);
        motion_.disconnect();
    }

    SafeStopGuard(const SafeStopGuard&) = delete;
    SafeStopGuard& operator=(const SafeStopGuard&) = delete;

private:
    bpx_sdk::MotionLevelControl& motion_;
};

bool waitUntilStanding(bpx_sdk::MotionLevelControl& motion) {
    std::cout << "Sending stand-up command; timeout="
              << kStandUpTimeout.count() << " seconds.\n";
    const auto deadline = std::chrono::steady_clock::now() + kStandUpTimeout;
    auto next_report = std::chrono::steady_clock::now();

    while (!g_stop_requested && std::chrono::steady_clock::now() < deadline) {
        std::uint8_t state = 0;
        if (motion.getCurrentMotionState(&state) &&
            state == static_cast<std::uint8_t>(bpx_sdk::MotionState::Motion)) {
            printState(motion);
            return true;
        }
        if (!motion.setStandUp()) {
            std::cerr << "failed to send stand-up command\n";
            return false;
        }
        if (std::chrono::steady_clock::now() >= next_report) {
            printState(motion);
            next_report += std::chrono::seconds(1);
        }
        std::this_thread::sleep_for(kPollInterval);
    }

    std::cerr << "robot did not report Motion state before timeout\n";
    return false;
}

void stopAndSwitchToWalk(bpx_sdk::MotionLevelControl& motion) {
    std::cout << "Stopping: send zero velocity and switch back to Walk.\n";
    motion.setVelocity(0.0f, 0.0f, 0.0f);
    motion.setWalk();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    disableVelocityControl(motion);
}

bool runVelocityPhase(bpx_sdk::MotionLevelControl& motion,
                      const char* description,
                      float speed_mps,
                      std::chrono::seconds duration,
                      bool* has_gait_feedback,
                      std::uint8_t* final_gait,
                      bool* has_sub_gait_feedback,
                      std::uint8_t* final_sub_gait) {
    std::cout << description << ": setVelocity(" << speed_mps
              << ", 0, 0) for " << duration.count() << " seconds.\n";
    if (!motion.setVelocity(speed_mps, 0.0f, 0.0f)) {
        std::cerr << "failed to send velocity command\n";
        return false;
    }

    const auto deadline = std::chrono::steady_clock::now() + duration;
    while (!g_stop_requested && std::chrono::steady_clock::now() < deadline) {
        std::uint8_t gait = 0;
        std::uint8_t sub_gait = 0;
        const bool has_gait = motion.getCurrentGait(&gait);
        const bool has_sub_gait = motion.getSubGait(&sub_gait);
        if (has_gait || has_sub_gait) {
            std::cout << description << " feedback: gait=";
            if (has_gait) {
                *has_gait_feedback = true;
                *final_gait = gait;
                std::cout << static_cast<unsigned int>(gait);
            } else {
                std::cout << "unknown";
            }
            std::cout << " sub_gait=";
            if (has_sub_gait) {
                *has_sub_gait_feedback = true;
                *final_sub_gait = sub_gait;
                std::cout
                    << static_cast<int>(static_cast<std::int8_t>(sub_gait));
            } else {
                std::cout << "unknown";
            }
            std::cout << '\n';
        }
        std::this_thread::sleep_for(kPollInterval);
    }
    return !g_stop_requested;
}

bool runMotionObservation(bpx_sdk::MotionLevelControl& motion,
                          float speed_mps) {
    motion.setVelocityControlFlag(true);

    bool has_gait_feedback = false;
    std::uint8_t final_gait = 0;
    bool has_sub_gait_feedback = false;
    std::uint8_t final_sub_gait = 0;
    if (!runVelocityPhase(motion,
                          "Phase 1/3, Bound forward",
                          speed_mps,
                          kForwardPhaseDuration,
                          &has_gait_feedback,
                          &final_gait,
                          &has_sub_gait_feedback,
                          &final_sub_gait) ||
        !runVelocityPhase(motion,
                          "Phase 2/3, pause in place",
                          0.0f,
                          kPausePhaseDuration,
                          &has_gait_feedback,
                          &final_gait,
                          &has_sub_gait_feedback,
                          &final_sub_gait) ||
        !runVelocityPhase(motion,
                          "Phase 3/3, Bound forward again",
                          speed_mps,
                          kForwardPhaseDuration,
                          &has_gait_feedback,
                          &final_gait,
                          &has_sub_gait_feedback,
                          &final_sub_gait)) {
        return false;
    }

    stopAndSwitchToWalk(motion);
    std::cout << "feedback summary after forward/pause/forward: final gait=";
    if (has_gait_feedback) {
        std::cout << static_cast<unsigned int>(final_gait)
                  << " (expected WalkPhase=6 for Bound)";
    } else {
        std::cout << "unknown";
    }
    std::cout << " sub_gait=";
    if (has_sub_gait_feedback) {
        std::cout << static_cast<int>(static_cast<std::int8_t>(final_sub_gait))
                  << " (expected Bound=1)\n";
    } else {
        std::cout << "unknown; use physical observation\n";
    }
    return !g_stop_requested;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseOptions(argc, argv, &options)) {
        return 2;
    }
    if (options.help_requested) {
        return 0;
    }

    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    const float forward_distance_m =
        options.speed_mps *
        static_cast<float>(2 * kForwardPhaseDuration.count());
    std::cout
        << "REAL ROBOT TEST: Bound forward, pause, then forward again\n"
        << "robot_ip=" << options.robot_ip
        << " speed=" << options.speed_mps << " m/s"
        << " phase durations=" << kBoundPreparationDuration.count()
        << "s Bound preparation + " << kForwardPhaseDuration.count()
        << "s forward + " << kPausePhaseDuration.count()
        << "s pause + " << kForwardPhaseDuration.count() << "s forward"
        << " total travel distance is approximately "
        << forward_distance_m << " m\n\n"
        << "Clear the test area, suspend the robot if required by your safety\n"
        << "procedure, and keep the emergency stop available.\n"
        << "Expected: Bound remains selected through the pause and restart.\n"
        << "Bound is reported as WalkPhase gait (6).\n";

    if (!confirm("Type RUN to connect and begin stand-up preparation.", "RUN")) {
        std::cout << "test cancelled\n";
        return 2;
    }

    bpx_sdk::MotionLevelControl motion;
    motion.setRobotIp(options.robot_ip.c_str());
    motion.setRobotStateUploadPort(options.state_port);
    motion.setTcpLocalPort(options.tcp_local_port);
    motion.setRobotStateUploadRate(100);
    motion.setMotionCommandRate(50);

    if (!motion.connect()) {
        std::cerr << "failed to connect to robot " << options.robot_ip << '\n';
        return 1;
    }
    SafeStopGuard safe_stop(motion);

    motion.setZeroPositionsFlag();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (!waitUntilStanding(motion)) {
        return 1;
    }

    const std::string move_prompt =
        "Robot is standing. Clear at least " +
        std::to_string(forward_distance_m + 1.0f) +
        " meters ahead, then type MOVE to run the Bound sequence.";
    if (!confirm(move_prompt, "MOVE")) {
        std::cout << "motion phase cancelled\n";
        return 2;
    }

    motion.setBound();
    std::cout << "Selected Bound gait in the SDK. Waiting "
              << kBoundPreparationDuration.count()
              << " seconds before sending forward velocity.\n";
    std::this_thread::sleep_for(kBoundPreparationDuration);
    if (g_stop_requested) {
        std::cerr << "test interrupted while waiting to move\n";
        return 1;
    }
    if (!runMotionObservation(motion, options.speed_mps)) {
        std::cerr << "test interrupted\n";
        return 1;
    }

    const bool passed = confirm(
        "Did the robot Bound forward, pause in place, and Bound forward again? "
        "Enter y for PASS; any other input means FAIL.",
        "y");
    std::cout << (passed ? "PASS" : "FAIL")
              << ": real-robot gait observation\n";
    return passed ? 0 : 1;
}
