#include "../hcde/common.hh"
#include "../hcde/fast_profile.inl"
#include "../hcde/strong_profile.inl"
#include "../hcde/cpu_encoder.inl"
#include "../hcde/mt_cpu_encoder.inl"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


using namespace hcde;
using namespace hcde::detail;


template<typename Arithmetic>
static std::vector<Arithmetic> make_random_vector(size_t size) {
    std::vector<Arithmetic> vector(size);
    auto gen = std::minstd_rand();
    if constexpr (std::is_floating_point_v<Arithmetic>) {
        auto dist = std::uniform_real_distribution<Arithmetic>();
        std::generate(vector.begin(), vector.end(), [&] { return dist(gen); });
    } else {
        auto dist = std::uniform_int_distribution<Arithmetic>();
        std::generate(vector.begin(), vector.end(), [&] { return dist(gen); });
    }
    return vector;
}


TEST_CASE("for_each_in_hypercube") {
    int data[10000];
    std::iota(std::begin(data), std::end(data), 0);

    SECTION("1d") {
        std::vector<int> indices;
        for_each_in_hypercube(slice<int, 1>(data, extent<1>(100)), extent<1>(5), 4,
                [&](int x) { indices.push_back(x); });
        CHECK(indices == std::vector{5, 6, 7, 8});
    }
    SECTION("2d") {
        std::vector<int> indices;
        for_each_in_hypercube(slice<int, 2>(data, extent<2>(10, 10)), extent<2>(1, 2), 3,
                [&](int x) { indices.push_back(x); });
        CHECK(indices == std::vector{12, 13, 14, 22, 23, 24, 32, 33, 34});
    }
    SECTION("3d") {
        std::vector<int> indices;
        for_each_in_hypercube(slice<int, 3>(data, extent<3>(10, 10, 10)), extent<3>(1, 0, 2), 2,
                [&](int x) { indices.push_back(x); });
        CHECK(indices == std::vector{102, 103, 112, 113, 202, 203, 212, 213});
    }
    SECTION("4d") {
        std::vector<int> indices;
        for_each_in_hypercube(slice<int, 4>(data, extent<4>(10, 10, 10, 10)),
                extent<4>(1, 0, 2, 3), 2, [&](int x) { indices.push_back(x); });
        CHECK(indices == std::vector{1023, 1024, 1033, 1034, 1123, 1124, 1133, 1134,
                2023, 2024, 2033, 2034, 2123, 2124, 2133, 2134});
    }
}


TEST_CASE("load_bits") {
    SECTION("for 8-bit integers") {
        alignas(uint8_t) const uint8_t bits[2] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint8_t>(const_bit_ptr<1>(bits, 0), 2) == 0b10);
        CHECK(load_bits<uint8_t>(const_bit_ptr<1>(bits, 2), 3) == 0b100);
        CHECK(load_bits<uint8_t>(const_bit_ptr<1>(bits, 5), 4) == 0b1001);
    }
    SECTION("for 16-bit integers") {
        alignas(uint16_t) const uint8_t bits[4] = {0b1010'0100, 0b1000'0000};
        CHECK(load_bits<uint16_t>(const_bit_ptr<2>(bits, 0), 2) == 0b10);
        CHECK(load_bits<uint16_t>(const_bit_ptr<2>(bits, 2), 3) == 0b100);
        CHECK(load_bits<uint16_t>(const_bit_ptr<2>(bits, 5), 4) == 0b1001);
    }
    SECTION("for 32-bit integers") {
        alignas(uint32_t) const uint8_t bits[8] = {0b1010'0100, 0b1100'0000};
        CHECK(load_bits<uint32_t>(const_bit_ptr<4>(bits, 0), 2) == 0b10);
        CHECK(load_bits<uint32_t>(const_bit_ptr<4>(bits, 2), 3) == 0b100);
        CHECK(load_bits<uint32_t>(const_bit_ptr<4>(bits, 5), 4) == 0b1001);
        CHECK(load_bits<uint32_t>(const_bit_ptr<4>(bits, 9), 27) == 0b100000000000000000000000000);
    }
    SECTION("for 64-bit integers") {
        alignas(uint64_t) const uint8_t bits[16] = {0b1010'0100, 0b1100'0000};
        CHECK(load_bits<uint64_t>(const_bit_ptr<8>(bits, 0), 2) == 0b10);
        CHECK(load_bits<uint64_t>(const_bit_ptr<8>(bits, 2), 3) == 0b100);
        CHECK(load_bits<uint64_t>(const_bit_ptr<8>(bits, 5), 4) == 0b1001);
        CHECK(load_bits<uint64_t>(const_bit_ptr<8>(bits, 9), 27) == 0b100000000000000000000000000);
    }
}

TEST_CASE("store_bits_linear") {
    SECTION("for 8-bit integers") {
        alignas(uint8_t) const uint8_t expected_bits[2] = {0b1010'0100, 0b1000'0000};
        alignas(uint8_t) uint8_t bits[2] = {0};
        store_bits_linear<uint8_t>(bit_ptr<1>(bits, 0), 2, 0b10);
        store_bits_linear<uint8_t>(bit_ptr<1>(bits, 2), 3, 0b100);
        store_bits_linear<uint8_t>(bit_ptr<1>(bits, 5), 4, 0b1001);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
    SECTION("for 32-bit integers") {
        alignas(uint16_t) const uint8_t expected_bits[4] = {0b1010'0100, 0b1000'0000};
        alignas(uint16_t) uint8_t bits[4] = {0};
        store_bits_linear<uint16_t>(bit_ptr<2>(bits, 0), 2, 0b10);
        store_bits_linear<uint16_t>(bit_ptr<2>(bits, 2), 3, 0b100);
        store_bits_linear<uint16_t>(bit_ptr<2>(bits, 5), 4, 0b1001);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
    SECTION("for 32-bit integers") {
        alignas(uint32_t) const uint8_t expected_bits[8] = {0b1010'0100, 0b1100'0000};
        alignas(uint32_t) uint8_t bits[8] = {0};
        store_bits_linear<uint32_t>(bit_ptr<4>(bits, 0), 2, 0b10);
        store_bits_linear<uint32_t>(bit_ptr<4>(bits, 2), 3, 0b100);
        store_bits_linear<uint32_t>(bit_ptr<4>(bits, 5), 4, 0b1001);
        store_bits_linear<uint32_t>(bit_ptr<4>(bits, 9), 27, 0b100000000000000000000000000);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
    SECTION("for 32-bit integers") {
        alignas(uint64_t) const uint8_t expected_bits[16] = {0b1010'0100, 0b1100'0000};
        alignas(uint64_t) uint8_t bits[16] = {0};
        store_bits_linear<uint64_t>(bit_ptr<8>(bits, 0), 2, 0b10);
        store_bits_linear<uint64_t>(bit_ptr<8>(bits, 2), 3, 0b100);
        store_bits_linear<uint64_t>(bit_ptr<8>(bits, 5), 4, 0b1001);
        store_bits_linear<uint64_t>(bit_ptr<8>(bits, 9), 27, 0b100000000000000000000000000);
        CHECK(memcmp(bits, expected_bits, sizeof bits) == 0);
    }
}


TEMPLATE_TEST_CASE("value en-/decoding reproduces bit-identical values", "[profile]",
        (fast_profile<float, 2>), (strong_profile<float, 2>))
{
    const auto input = make_random_vector<float>(100);

    TestType p;
    std::vector<uint32_t> bits(input.size());
    for (unsigned i = 0; i < input.size(); ++i) {
        bits[i] = p.load_value(&input[i]);
    }

    std::vector<float> output(input.size());
    for (unsigned i = 0; i < output.size(); ++i) {
        p.store_value(&output[i], bits[i]);
    }
    CHECK(memcmp(input.data(), output.data(), input.size() * sizeof(float)) == 0);
}


TEMPLATE_TEST_CASE("block en-/decoding reproduces bit-identical values", "[profile]",
        (fast_profile<float, 2>), (strong_profile<float, 2>))
{
    const auto random = make_random_vector<uint32_t>(ipow(TestType::hypercube_side_length, 2));
    auto halves = random; for (auto &h: halves) { h >>= 16u; };
    const auto zeroes = std::vector<uint32_t>(random.size(), 0);

    const auto test_vector = [](const std::vector<uint32_t> &input) {
        TestType p;
        std::vector<uint32_t> output(input.size());
        // stream buffer must be large enough for aligned stores. TODO can this be expressed generically?
        std::byte stream[TestType::compressed_block_size_bound + sizeof(uint32_t)];
        memset(stream, 0, sizeof stream);
        p.encode_block(input.data(), stream);
        p.decode_block(stream, output.data());

        CHECK(memcmp(input.data(), output.data(), input.size() * sizeof(uint32_t)) == 0);
    };

    test_vector(random);
    test_vector(halves);
    test_vector(zeroes);
}


using border_slice = std::pair<size_t, size_t>;
using slice_vec = std::vector<border_slice>;

namespace std {
    ostream &operator<<(ostream &os, const border_slice &s) {
        return os << "(" << s.first << ", " << s.second << ")";
    }
}

template<unsigned Dims>
static auto dump_border_slices(const extent<Dims> &size, unsigned side_length) {
    slice_vec v;
    for_each_border_slice(size, side_length, [&](size_t offset, size_t count) {
        v.emplace_back(offset, count);
    });
    return v;
}


TEST_CASE("for_each_border_slice iterates correctly") {
    CHECK(dump_border_slices(extent<2>{4, 4}, 4) == slice_vec{});
    CHECK(dump_border_slices(extent<2>{4, 6}, 2) == slice_vec{});
    CHECK(dump_border_slices(extent<2>{5, 4}, 4) == slice_vec{{16, 4}});
    CHECK(dump_border_slices(extent<2>{4, 5}, 4) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(extent<2>{4, 5}, 2) == slice_vec{{4, 1}, {9, 1}, {14, 1}, {19, 1}});
    CHECK(dump_border_slices(extent<2>{4, 6}, 4) == slice_vec{{4, 2}, {10, 2}, {16, 2}, {22, 2}});
    CHECK(dump_border_slices(extent<2>{4, 6}, 5) == slice_vec{{0, 24}});
    CHECK(dump_border_slices(extent<2>{6, 4}, 5) == slice_vec{{0, 24}});
}


struct test_profile {
    using data_type = float;
    using bits_type = uint32_t;
    using hypercube_offset_type = uint32_t;

    constexpr static unsigned dimensions = 2;
    constexpr static unsigned hypercube_side_length = 4;
    constexpr static size_t compressed_block_size_bound
            = sizeof(bits_type) * ipow(hypercube_side_length, dimensions);

    size_t encode_block(const bits_type *bits, void *stream) const {
        size_t n_bytes = ipow(hypercube_side_length, dimensions) * sizeof(bits_type);
        memcpy(stream, bits, n_bytes);
        return n_bytes;
    }

    size_t decode_block(const void *stream, bits_type *bits) const {
        size_t n_bytes = ipow(hypercube_side_length, dimensions) * sizeof(bits_type);
        memcpy(bits, stream, n_bytes);
        return n_bytes;
    }

    bits_type load_value(const data_type *data) const {
        bits_type bits;
        memcpy(&bits, data, sizeof bits);
        return bits;
    }

    void store_value(data_type *data, bits_type bits) const {
        memcpy(data, &bits, sizeof bits);
    }
};

TEST_CASE("file produces a sane superblock / hypercube / header layout", "[file]") {
    const size_t n = 100;
    const auto n_hypercubes_per_dim = n / test_profile::hypercube_side_length;

    std::vector<std::vector<extent<test_profile::dimensions>>> superblocks;
    std::vector<bool> visited(ipow(n_hypercubes_per_dim, 2));

    file<test_profile> f(extent<2>{n, n});
    f.for_each_superblock([&](auto sb) {
        std::vector<extent<test_profile::dimensions>> blocks;
        sb.for_each_hypercube([&](auto off) {
            CHECK(off[0] < n);
            CHECK(off[1] < n);
            CHECK(off[0] % test_profile::hypercube_side_length == 0);
            CHECK(off[1] % test_profile::hypercube_side_length == 0);

            auto idx = off[0] / test_profile::hypercube_side_length * n_hypercubes_per_dim
                + off[1] / test_profile::hypercube_side_length;
            CHECK(!visited[idx]);
            visited[idx] = true;

            blocks.push_back(off);
        });
        CHECK(blocks.size() == sb.num_hypercubes());
        superblocks.push_back(std::move(blocks));
    });

    CHECK(std::all_of(visited.begin(), visited.end(), [](auto b) { return b; }));

    CHECK(superblocks.size() == f.num_superblocks());
    CHECK(!superblocks.empty());

    CHECK(f.file_header_length() == f.num_superblocks() * sizeof(uint64_t));
    CHECK(f.num_hypercubes() == ipow(n_hypercubes_per_dim, 2));
}


TEMPLATE_TEST_CASE("encoder produces the expected bit stream", "[encoder]",
        (cpu_encoder<test_profile>), (mt_cpu_encoder<test_profile>))
{
    const size_t n = 199;
    const auto cell = 3.141592f;
    const auto border = 2.71828f;
    std::vector<float> data(n * n);
    slice<float, 2> array(data.data(), extent{n, n});

    const auto border_start = n / test_profile::hypercube_side_length * test_profile::hypercube_side_length;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            array[{i, j}] = i < border_start && j < border_start ? cell : border;
        }
    }

    file<test_profile> f(array.size());
    REQUIRE(f.num_superblocks() > 1);

    TestType encoder;
    std::vector<std::byte> stream(encoder.compressed_size_bound(array.size()));
    size_t size = encoder.compress(array, stream.data());

    CHECK(size <= stream.size());
    stream.resize(size);

    const size_t hypercube_size = sizeof(float) * ipow(test_profile::hypercube_side_length, 2);

    const auto *file_header = stream.data();
    size_t superblock_index = 0;
    size_t current_superblock_offset = f.file_header_length();
    f.for_each_superblock([&](auto superblock) {
        if (superblock_index > 0) {
            const void *file_offset_address = file_header + (superblock_index - 1) * sizeof(uint64_t);
            CHECK(endian_transform(load_unaligned<uint64_t>(file_offset_address)) == current_superblock_offset);
        }

        const auto *superblock_header = stream.data() + current_superblock_offset;
        size_t hypercube_index = 0;
        superblock.for_each_hypercube([&](auto) {
            const auto hypercube_offset = hypercube_index * hypercube_size;
            if (hypercube_index > 0) {
                const void *superblock_offset_address = superblock_header
                        + (hypercube_index - 1) * sizeof(test_profile::hypercube_offset_type);
                CHECK(endian_transform(load_unaligned<test_profile::hypercube_offset_type>(superblock_offset_address))
                      == hypercube_offset);
            }
            for (size_t i = 0; i < ipow(test_profile::hypercube_side_length, 2); ++i) {
                float value;
                const void *value_offset_address = superblock_header + f.superblock_header_length() + i * sizeof value;
                test_profile{}.store_value(&value, load_unaligned<test_profile::bits_type>(value_offset_address));
                CHECK(memcmp(&value, &cell, sizeof value) == 0);
            }
            ++hypercube_index;
        });

        ++superblock_index;
        current_superblock_offset += f.superblock_header_length() + superblock.num_hypercubes() * hypercube_size;
    });

    const void *border_offset_address = file_header + (f.num_superblocks() - 1) * sizeof(uint64_t);
    CHECK(endian_transform(load_unaligned<uint64_t>(border_offset_address)) == current_superblock_offset);
    size_t n_border_elems = 0;
    for_each_border_slice(array.size(), test_profile::hypercube_side_length, [&](auto, auto count) {
        for (unsigned i = 0; i < count; ++i) {
            float value;
            const void *value_offset_address = stream.data() + current_superblock_offset
                    + (n_border_elems + i) * sizeof value;
            test_profile{}.store_value(&value, load_unaligned<test_profile::bits_type>(value_offset_address));
            CHECK(memcmp(&value, &border, sizeof value) == 0);
        }
        n_border_elems += count;
    });
    CHECK(n_border_elems == array.size().linear_offset() - ipow(border_start, 2));
}

TEMPLATE_TEST_CASE("encoder reproduces the bit-identical array", "[encoder]",
        (cpu_encoder<test_profile>), (mt_cpu_encoder<test_profile>))
{
    const size_t n = 199;
    auto input_data = make_random_vector<float>(n * n);
    slice<const float, 2> input(input_data.data(), extent{n, n});

    TestType encoder;
    std::vector<std::byte> stream(encoder.compressed_size_bound(input.size()));
    stream.resize(encoder.compress(input, stream.data()));

    std::vector<float> output_data(n * n);
    slice<float, 2> output(output_data.data(), extent{n, n});
    encoder.decompress(stream.data(), stream.size(), output);

    CHECK(memcmp(input_data.data(), output_data.data(), input_data.size() * sizeof(float)) == 0);
}
