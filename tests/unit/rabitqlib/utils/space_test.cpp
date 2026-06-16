#include <gtest/gtest.h>
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/defines.hpp"
#include "test_helpers.hpp"
#include "test_data.hpp"
#include <vector>
#include <cmath>

using namespace rabitqlib;
using namespace rabitq_test;

namespace {

uint64_t ReferencePlaneMask16(const uint16_t* values, size_t bit_idx) {
    uint64_t mask = 0;
    for (size_t lane = 0; lane < 64; ++lane) {
        if ((values[lane] >> bit_idx) & 0x1U) {
            mask |= 1ULL << (63 - lane);
        }
    }
    return mask;
}

uint64_t ReferencePlaneMask8(const uint8_t* values, size_t bit_idx) {
    uint64_t mask = 0;
    for (size_t lane = 0; lane < 64; ++lane) {
        if ((values[lane] >> bit_idx) & 0x1U) {
            mask |= 1ULL << (63 - lane);
        }
    }
    return mask;
}

}  // namespace

TEST(Select_IP_Func, returns_correct_function_pointer) {
    auto ip_func = select_excode_ipfunc(0);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip16_fxu1_avx);

    ip_func = select_excode_ipfunc(1);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip16_fxu1_avx);

    ip_func = select_excode_ipfunc(2);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu2_avx);
    
    ip_func = select_excode_ipfunc(3);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu3_avx);

    ip_func = select_excode_ipfunc(4);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip16_fxu4_avx);

    ip_func = select_excode_ipfunc(5);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu5_avx);

    ip_func = select_excode_ipfunc(6);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu6_avx);

    ip_func = select_excode_ipfunc(7);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, excode_ipimpl::ip64_fxu7_avx);

    ip_func = select_excode_ipfunc(8);
    ASSERT_NE(ip_func, nullptr);
    ASSERT_EQ(ip_func, (excode_ipimpl::ip_fxi<float, uint8_t>));
}

TEST(ip16_fxu1_avx, ip_works) {
    srand(42);
    size_t dim = 64;
    float query[dim];
    uint8_t codes[dim/8];
    
    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    for (size_t i = 0; i < dim / 8; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }

    ASSERT_NEAR(rabitqlib::excode_ipimpl::ip16_fxu1_avx(query, codes, dim), 15055.81f, 0.1f);
}

TEST(ip64_fxu2_avx, ip_works) {
    srand(42);
    size_t dim = 64*4;
    float query[dim];
    uint8_t codes[dim/4];
    
    for (size_t i = 0; i < dim; ++i) {
        query[i] = static_cast<float>(rand()) / RAND_MAX * 1000.0f;
    }

    for (size_t i = 0; i < dim / 4; ++i) {
        codes[i] = static_cast<uint8_t>(rand() % 256);
    }
    ASSERT_NEAR(rabitqlib::excode_ipimpl::ip64_fxu2_avx(query, codes, dim), 217584.15f, 0.1f);
}

TEST(TransposeBin, ProducesExpectedBitPlanes) {
    constexpr size_t kPaddedDim = 64;
    constexpr size_t kBits = 4;
    uint16_t values[kPaddedDim];
    uint64_t transposed[kBits] = {};

    for (size_t i = 0; i < kPaddedDim; ++i) {
        values[i] = static_cast<uint16_t>((i * 3) & 0x0f);
    }

    rabitqlib::new_transpose_bin(values, transposed, kPaddedDim, kBits);

    for (size_t bit_idx = 0; bit_idx < kBits; ++bit_idx) {
        ASSERT_EQ(transposed[bit_idx], ReferencePlaneMask16(values, bit_idx));
    }
}

TEST(TransposeBin512, ProducesExpectedBitPlanesAcrossChunks) {
    constexpr size_t kPaddedDim = 640;
    constexpr size_t kBits = 4;
    std::vector<uint8_t> values(kPaddedDim);
    std::vector<uint64_t> transposed((kPaddedDim / 64) * kBits);

    for (size_t i = 0; i < kPaddedDim; ++i) {
        values[i] = static_cast<uint8_t>((i * 5) & 0x0f);
    }

    rabitqlib::new_transpose_bin_512(values.data(), transposed.data(), kPaddedDim, kBits);

    size_t output_offset = 0;
    for (size_t block_start = 0; block_start < kPaddedDim;) {
        size_t block_size = std::min<size_t>(512, kPaddedDim - block_start);
        size_t num_chunks = block_size / 64;

        for (size_t bit_idx = 0; bit_idx < kBits; ++bit_idx) {
            for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
                const uint8_t* chunk_values = values.data() + block_start + (chunk * 64);
                ASSERT_EQ(
                    transposed[output_offset + (bit_idx * num_chunks) + chunk],
                    ReferencePlaneMask8(chunk_values, bit_idx)
                );
            }
        }

        block_start += block_size;
        output_offset += num_chunks * kBits;
    }
}

TEST(MaskIpX0Q, MatchesOldImplementation) {
    constexpr size_t kPaddedDim = 128;
    float query[kPaddedDim];
    uint64_t data[kPaddedDim / 64];

    srand(42);
    for (size_t i = 0; i < kPaddedDim; ++i) {
        query[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }
    for (size_t i = 0; i < (kPaddedDim / 64); ++i) {
        uint64_t hi = static_cast<uint64_t>(rand());
        uint64_t lo = static_cast<uint64_t>(rand());
        data[i] = (hi << 32) | lo;
    }

    ASSERT_FLOAT_EQ(
        rabitqlib::mask_ip_x0_q(query, data, kPaddedDim),
        rabitqlib::mask_ip_x0_q_old(query, data, kPaddedDim)
    );
}