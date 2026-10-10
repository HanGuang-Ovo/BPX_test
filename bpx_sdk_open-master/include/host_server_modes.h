#ifndef BPX_SDK_HOST_SERVER_MODES_H_
#define BPX_SDK_HOST_SERVER_MODES_H_

#include <cstdint>

namespace bpx_protocol {

// SubscribeStateResp.reserved[0] identifies control-mode feedback revision 1;
// reserved[1] carries 0=remote control, 1=navigator, 255=unknown.
constexpr uint32_t kControlModeFeedbackTag = 0x434D0001U;

enum class HostServerMode : uint8_t {
    Test = 0,
    StateQuery = 1,
    MotionControl = 2,
    JointControl = 3,
    VersionQuery = 4,
    TimeSync = 5,
    IdentityQuery = 6,
};

constexpr uint8_t ToWireValue(HostServerMode mode) {
    return static_cast<uint8_t>(mode);
}

}  // namespace bpx_protocol

#endif  // BPX_SDK_HOST_SERVER_MODES_H_
