#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"

namespace {

constexpr std::array<size_t, 6> kDims = {64, 128, 256, 768, 960, 1024};
constexpr size_t kIterations = 100;

void reference_accumulate_hacc(
    const uint8_t* codes,
    const uint16_t* lut,
    int32_t* result,
    size_t dim
) {
    for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
        result[i] = 0;
    }

    size_t num_codebook = dim >> 2;
    for (size_t m = 0; m < num_codebook; ++m) {
        const uint8_t* code_ptr = codes + (m * 16);
        const uint16_t* lut_ptr = lut + (m * 16);
        for (size_t j = 0; j < 16; ++j) {
            uint8_t packed_code = code_ptr[j];
            size_t permuted_index = static_cast<size_t>(rabitqlib::fastscan::kPerm0[j]);
            result[permuted_index] += lut_ptr[packed_code & 0x0f];
            result[permuted_index + 16] += lut_ptr[packed_code >> 4];
        }
    }
}

}  // namespace

TEST(HighAccFastscanTest, CurrentBackendMatchesReference) {
    std::mt19937 rng(987654321u);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> lut_dist(0, 65535);

    for (size_t dim : kDims) {
        SCOPED_TRACE(::testing::Message() << "dim=" << dim);

        std::vector<uint8_t> codes(dim * 4);
        std::vector<uint16_t> lut(dim * 4);
        std::vector<uint8_t> hc_lut(dim * 8);

        for (size_t iter = 0; iter < kIterations; ++iter) {
            for (auto& value : codes) {
                value = static_cast<uint8_t>(byte_dist(rng));
            }
            for (auto& value : lut) {
                value = static_cast<uint16_t>(lut_dist(rng));
            }

            rabitqlib::fastscan::transfer_lut_hacc(lut.data(), dim, hc_lut.data());

            std::array<int32_t, rabitqlib::fastscan::kBatchSize> expected{};
            std::array<int32_t, rabitqlib::fastscan::kBatchSize> actual{};

            reference_accumulate_hacc(codes.data(), lut.data(), expected.data(), dim);
            rabitqlib::fastscan::accumulate_hacc(codes.data(), hc_lut.data(), actual.data(), dim);

            for (size_t i = 0; i < rabitqlib::fastscan::kBatchSize; ++i) {
                ASSERT_EQ(actual[i], expected[i])
                    << "dim=" << dim << ", iteration=" << iter << ", slot=" << i;
            }
        }
    }
}