#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "rabitqlib/fastscan/fastscan.hpp"

namespace {

constexpr std::array<size_t, 6> kDims = {64, 128, 256, 768, 960, 1024};
constexpr size_t kIterations = 100;

const char* backend_name() {
#if defined(__AVX512BW__) || defined(__AVX2__)
    return "x86";
#elif defined(__ARM_FEATURE_SVE)
    return "sve";
#else
    return "scalar";
#endif
}

}  // namespace

TEST(FastscanAccumulateTest, CurrentBackendMatchesScalarReference) {
    std::mt19937 rng(123456789u);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    for (size_t dim : kDims) {
        SCOPED_TRACE(::testing::Message() << "dim=" << dim << ", backend=" << backend_name());

        std::vector<uint8_t> codes(dim * 4);
        std::vector<uint8_t> lut(dim * 4);

        for (size_t iter = 0; iter < kIterations; ++iter) {
            for (auto& value : codes) {
                value = static_cast<uint8_t>(byte_dist(rng));
            }
            for (auto& value : lut) {
                value = static_cast<uint8_t>(byte_dist(rng));
            }

            std::array<uint16_t, rabitqlib::fastscan::kBatchSize> expected{};
            std::array<uint16_t, rabitqlib::fastscan::kBatchSize> actual{};

            rabitqlib::fastscan::scalar::accumulate(
                codes.data(), lut.data(), expected.data(), dim
            );
            rabitqlib::fastscan::accumulate(codes.data(), lut.data(), actual.data(), dim);

            for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
                ASSERT_EQ(actual[i], expected[i])
                    << "backend=" << backend_name() << ", dim=" << dim
                    << ", iteration=" << iter << ", slot=" << i;
            }
        }
    }
}