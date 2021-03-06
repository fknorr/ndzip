#include <ndzip/common.hh>
#include <ndzip/cpu_encoder.inl>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


using namespace ndzip;
using namespace ndzip::detail;


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


TEMPLATE_TEST_CASE("block transform is reversible", "[profile]", (profile<float, 1>),
        (profile<float, 2>), (profile<float, 3>), (profile<double, 1>), (profile<double, 2>),
        (profile<double, 3>) ) {
    using bits_type = typename TestType::bits_type;

    const auto input = make_random_vector<bits_type>(
            ipow(TestType::hypercube_side_length, TestType::dimensions));

    auto transformed = input;
    detail::block_transform(
            transformed.data(), TestType::dimensions, TestType::hypercube_side_length);

    detail::inverse_block_transform(
            transformed.data(), TestType::dimensions, TestType::hypercube_side_length);

    CHECK(input == transformed);
}


TEMPLATE_TEST_CASE("CPU zero-word compaction is reversible", "[cpu]", uint32_t, uint64_t) {
    std::vector<TestType> input(bitsof<TestType>);
    auto gen = std::minstd_rand();  // NOLINT(cert-msc51-cpp)
    auto dist = std::uniform_int_distribution<TestType>();
    size_t zeroes = 0;
    for (auto &v : input) {
        auto r = dist(gen);
        if (r % 5 == 2) {
            v = 0;
            zeroes++;
        } else {
            v = r;
        }
    }

    std::vector<std::byte> compact((bitsof<TestType> + 1) * sizeof(TestType));
    auto bytes_written = detail::cpu::compact_zero_words(input.data(), compact.data());
    CHECK(bytes_written == (bitsof<TestType> + 1 - zeroes) * sizeof(TestType));

    std::vector<TestType> output(bitsof<TestType>);
    auto bytes_read = detail::cpu::expand_zero_words(compact.data(), output.data());
    CHECK(bytes_read == bytes_written);
    CHECK(output == input);
}


TEMPLATE_TEST_CASE("CPU bit transposition is reversible", "[cpu]", uint32_t, uint64_t) {
    alignas(cpu::simd_width_bytes) TestType input[bitsof<TestType>];
    auto rng = std::minstd_rand(1);
    auto bit_dist = std::uniform_int_distribution<TestType>();
    auto shift_dist = std::uniform_int_distribution<unsigned>(0, bitsof<TestType> - 1);
    for (auto &value : input) {
        value = bit_dist(rng) >> shift_dist(rng);
    }

    alignas(cpu::simd_width_bytes) TestType transposed[bitsof<TestType>];
    detail::cpu::transpose_bits(input, transposed);

    alignas(cpu::simd_width_bytes) TestType output[bitsof<TestType>];
    detail::cpu::transpose_bits(transposed, output);

    CHECK(memcmp(input, output, sizeof input) == 0);
}


using border_slice = std::pair<size_t, size_t>;
using slice_vec = std::vector<border_slice>;

namespace std {
ostream &operator<<(ostream &os, const border_slice &s) {
    return os << "(" << s.first << ", " << s.second << ")";
}
}  // namespace std

template<unsigned Dims>
static auto dump_border_slices(const extent<Dims> &size, unsigned side_length) {
    slice_vec v;
    for_each_border_slice(
            size, side_length, [&](size_t offset, size_t count) { v.emplace_back(offset, count); });
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


TEMPLATE_TEST_CASE("file produces a sane hypercube / header layout", "[file]",
        (std::integral_constant<unsigned, 1>), (std::integral_constant<unsigned, 2>),
        (std::integral_constant<unsigned, 3>), (std::integral_constant<unsigned, 4>) ) {
    constexpr unsigned dims = TestType::value;
    using profile = detail::profile<float, dims>;
    const size_t n = 100;
    const auto n_hypercubes_per_dim = n / profile::hypercube_side_length;
    const auto side_length = profile::hypercube_side_length;

    extent<dims> size;
    for (unsigned d = 0; d < dims; ++d) {
        size[d] = n;
    }

    std::vector<std::vector<extent<dims>>> superblocks;
    std::vector<bool> visited(ipow(n_hypercubes_per_dim, dims));

    file<profile> f(size);
    std::vector<extent<dims>> blocks;
    size_t hypercube_index = 0;
    f.for_each_hypercube([&](auto hc_offset, auto hc_index) {
        CHECK(hc_index == hypercube_index);

        auto off = hc_offset;
        for (unsigned d = 0; d < dims; ++d) {
            CHECK(off[d] < n);
            CHECK(off[d] % side_length == 0);
        }

        auto cell_index = off[0] / side_length;
        for (unsigned d = 1; d < dims; ++d) {
            cell_index = cell_index * n_hypercubes_per_dim + off[d] / side_length;
        }
        CHECK(!visited[cell_index]);
        visited[cell_index] = true;

        blocks.push_back(off);
        ++hypercube_index;
    });
    CHECK(blocks.size() == f.num_hypercubes());

    CHECK(std::all_of(visited.begin(), visited.end(), [](auto b) { return b; }));

    CHECK(f.file_header_length() == f.num_hypercubes() * sizeof(detail::file_offset_type));
    CHECK(f.num_hypercubes() == ipow(n_hypercubes_per_dim, dims));
}


/* requires Profile parameters
TEMPLATE_TEST_CASE("encoder produces the expected bit stream", "[encoder]",
    (cpu_encoder<float, 2>), (cpu_encoder<float, 3>),
    (mt_cpu_encoder<float, 2>), (mt_cpu_encoder<float, 3>)
) {
    using profile = detail::profile<typename TestType::data_type, TestType::dimensions>;
    using bits_type = typename profile::bits_type;
    using hc_offset_type = typename profile::hypercube_offset_type;

    const size_t n = 199;
    const auto cell = 3.141592f;
    const auto border = 2.71828f;
    constexpr auto dims = profile::dimensions;

    std::vector<float> data(ipow(n, dims));
    slice<float, dims> array(data.data(), extent<dims>::broadcast(n));

    const auto border_start = n / profile::hypercube_side_length * profile::hypercube_side_length;
    for_each_in_hyperslab(array.size(), [=](auto index) {
        if (std::all_of(index.begin(), index.end(), [=](auto s) { return s < border_start; })) {
            array[index] = cell;
        } else {
            array[index] = border;
        }
    });

    file<profile> f(array.size());
    REQUIRE(f.num_hypercubes() > 1);

    TestType encoder;
    std::vector<std::byte> stream(ndzip::compressed_size_bound<data_type>(array.size()));
    size_t size = encoder.compress(array, stream.data());

    CHECK(size <= stream.size());
    stream.resize(size);

    const size_t hc_size = sizeof(float) * ipow(profile::hypercube_side_length, dims);

    const auto *file_header = stream.data();
    f.for_each_hypercube([&](auto hc, auto hc_index) {
        const auto hc_offset = hc_index * hc_size;
        if (hc_index > 0) {
            const void *hc_offset_address = file_header + (hc_index - 1) * sizeof(hc_offset_type);
            CHECK(endian_transform(load_unaligned<hc_offset_type>(hc_offset_address))
                == hc_offset);
        }
        for (size_t i = 0; i < ipow(profile::hypercube_side_length, dims); ++i) {
            float value;
            const void *value_offset_address = file_header + hc_offset + i * sizeof value;
            detail::store_value<profile>(&value, load_unaligned<bits_type>(value_offset_address));
            CHECK(memcmp(&value, &cell, sizeof value) == 0);
        }
    });

    const auto border_offset = f.file_header_length() + f.num_hypercubes() * hc_size;
    const void *border_offset_address = file_header + (f.num_hypercubes() - 1) * sizeof(uint64_t);
    CHECK(endian_transform(load_unaligned<uint64_t>(border_offset_address)) == border_offset);
    size_t n_border_elems = 0;
    for_each_border_slice(array.size(), profile::hypercube_side_length, [&](auto, auto count) {
        for (unsigned i = 0; i < count; ++i) {
            float value;
            const void *value_offset_address = stream.data() + border_offset
                + (n_border_elems + i) * sizeof value;
            detail::store_value<profile>(&value, load_unaligned<bits_type>(value_offset_address));
            CHECK(memcmp(&value, &border, sizeof value) == 0);
        }
        n_border_elems += count;
    });
    CHECK(n_border_elems == num_elements(array.size()) - ipow(border_start, dims));
}
*/


TEMPLATE_TEST_CASE("encoder reproduces the bit-identical array", "[encoder]",
        (cpu_encoder<float, 1>), (cpu_encoder<float, 2>), (cpu_encoder<float, 3>),
        (cpu_encoder<double, 1>), (cpu_encoder<double, 2>), (cpu_encoder<double, 3>)
#if NDZIP_OPENMP_SUPPORT
                                                                    ,
        (mt_cpu_encoder<float, 1>), (mt_cpu_encoder<float, 2>), (mt_cpu_encoder<float, 3>),
        (mt_cpu_encoder<double, 1>), (mt_cpu_encoder<double, 2>), (mt_cpu_encoder<double, 3>)
#endif
) {
    using data_type = typename TestType::data_type;
    using profile = detail::profile<data_type, TestType::dimensions>;

    constexpr auto dims = profile::dimensions;
    constexpr auto side_length = profile::hypercube_side_length;
    const size_t n = side_length * 4 - 1;

    auto input_data = make_random_vector<data_type>(ipow(n, dims));

    // Regression test: trigger bug in decoder optimization by ensuring first chunk = 0
    std::fill(input_data.begin(), input_data.begin() + bitsof<data_type>, data_type{});

    slice<const data_type, dims> input(input_data.data(), extent<dims>::broadcast(n));

    TestType encoder;
    std::vector<std::byte> stream(
            ndzip::compressed_size_bound<typename TestType::data_type>(input.size()));
    stream.resize(encoder.compress(input, stream.data()));

    std::vector<data_type> output_data(ipow(n, dims));
    slice<data_type, dims> output(output_data.data(), extent<dims>::broadcast(n));
    encoder.decompress(stream.data(), stream.size(), output);

    CHECK(memcmp(input_data.data(), output_data.data(), input_data.size() * sizeof(float)) == 0);
}

/* requires Profile parameters
TEST_CASE("load hypercube in GPU warp", "[gpu]") {
    SECTION("1d array") {
        std::vector<uint32_t> array{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::vector<uint32_t> buffer(4);
        hypercube<profile<uint32_t, 1>> hc(extent<1>{2});
        for (unsigned tid = 0; tid < NDZIP_WARP_SIZE; ++tid) {
            load_hypercube_warp(tid, hc, slice<uint32_t, 1>(array.data(), array.size()),
                    buffer.data());
        }
        CHECK(buffer == std::vector<uint32_t>{3, 4, 0, 0});
    }

    SECTION("2d array") {
        std::vector<uint32_t> array{
            10, 20, 30, 40, 50, 60, 70, 80, 90,
            11, 21, 31, 41, 51, 61, 71, 81, 91,
            12, 22, 32, 42, 52, 62, 72, 82, 92,
            13, 23, 33, 43, 53, 63, 73, 83, 93,
            14, 24, 34, 44, 54, 64, 74, 84, 94,
            15, 25, 35, 45, 55, 65, 75, 85, 95,
            16, 26, 36, 46, 56, 66, 76, 86, 96,
            17, 27, 37, 47, 57, 67, 77, 87, 97,
        };
        std::vector<uint32_t> buffer(6);
        hypercube<profile<uint32_t, 2>> hc(extent<2>{2, 3});
        for (unsigned tid = 0; tid < NDZIP_WARP_SIZE; ++tid) {
            load_hypercube_warp(tid, hc, slice<uint32_t, 2>(array.data(), extent{8, 9}),
                    buffer.data());
        }
        CHECK(buffer == std::vector<uint32_t>{42, 52, 43, 53, 0, 0});
    }

    SECTION("3d array") {
        std::vector<uint32_t> array{
            111, 211, 311, 411, 511,
            121, 221, 321, 421, 521,
            131, 231, 331, 431, 531,
            141, 241, 341, 441, 541,
            151, 251, 351, 451, 551,

            112, 212, 312, 412, 512,
            122, 222, 322, 422, 522,
            132, 232, 332, 432, 532,
            142, 242, 342, 442, 542,
            152, 252, 352, 452, 552,

            113, 213, 313, 413, 513,
            123, 223, 323, 423, 523,
            133, 233, 333, 433, 533,
            143, 243, 343, 443, 543,
            153, 253, 353, 453, 553,

            114, 214, 314, 414, 514,
            124, 224, 324, 424, 524,
            134, 234, 334, 434, 534,
            144, 244, 344, 444, 544,
            154, 254, 354, 454, 554,

            115, 215, 315, 415, 515,
            125, 225, 325, 425, 525,
            135, 235, 335, 435, 535,
            145, 245, 345, 445, 545,
            155, 255, 355, 455, 555,
        };
        std::vector<uint32_t> buffer(10);
        hypercube<profile<uint32_t, 3>> hc(extent<3>{1, 2, 3});
        for (unsigned tid = 0; tid < NDZIP_WARP_SIZE; ++tid) {
            load_hypercube_warp(tid, hc, slice<uint32_t, 3>(array.data(), extent{5, 5, 5}),
                    buffer.data());
        }
        CHECK(buffer == std::vector<uint32_t>{332, 432, 532, 342, 442, 542, 352, 452, 552, 0, 0});
    }
}
*/
