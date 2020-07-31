#pragma once

#include "common.hh"

#include <vector>


template<typename Profile>
size_t hcde::cpu_encoder<Profile>::compressed_size_bound(const extent<dimensions> &size) const {
    detail::file<Profile> file(size);
    size_t bound = file.combined_length_of_all_headers();
    bound += file.num_hypercubes() * Profile::compressed_block_size_bound;
    bound += detail::border_element_count(size, Profile::hypercube_side_length) * sizeof(data_type);
    return bound;
}


template<typename Profile>
size_t hcde::cpu_encoder<Profile>::compress(const slice<const data_type, dimensions> &data, void *stream) const {
    using bits_type = typename Profile::bits_type;

    constexpr static auto side_length = Profile::hypercube_side_length;
    detail::file<Profile> file(data.size());
    size_t stream_pos = file.file_header_length();

    std::vector<bits_type> cube;
    file.for_each_superblock([&](auto sb, auto sb_index) {
        size_t superblock_header_pos = stream_pos;
        stream_pos += file.superblock_header_length();

        Profile p;
        cube.resize(detail::ipow(side_length, Profile::dimensions) * sb.num_hypercubes());
        bits_type *cube_pos = cube.data();
        sb.for_each_hypercube([&](auto hc) {
            hc.for_each_cell(data, [&](auto *cell) { *cube_pos++ = p.load_value(cell); });
        });

        cube_pos = cube.data();
        sb.for_each_hypercube([&](auto, auto hc_index) {
            if (hc_index > 0) {
                auto offset_address = static_cast<char *>(stream) + superblock_header_pos
                    + (hc_index - 1) * sizeof(typename Profile::hypercube_offset_type);
                auto offset = static_cast<typename Profile::hypercube_offset_type>(
                    stream_pos - superblock_header_pos);
                detail::store_unaligned(offset_address, detail::endian_transform(offset));
            }
            stream_pos += p.encode_block(cube_pos, static_cast<char *>(stream) + stream_pos);
            cube_pos += detail::ipow(side_length, Profile::dimensions);
        });

        if (sb_index > 0) {
            auto file_offset_address = static_cast<char *>(stream)
                + (sb_index - 1) * sizeof(uint64_t);
            auto file_offset = static_cast<uint64_t>(superblock_header_pos);
            detail::store_unaligned(file_offset_address, detail::endian_transform(file_offset));
        }
    });

    auto border_offset_address = static_cast<char *>(stream) + (file.num_superblocks() - 1) * sizeof(uint64_t);
    detail::store_unaligned(border_offset_address, detail::endian_transform(stream_pos));
    stream_pos += detail::pack_border(static_cast<char *>(stream) + stream_pos, data, side_length);
    return stream_pos;
}


template<typename Profile>
size_t hcde::cpu_encoder<Profile>::decompress(const void *stream, size_t bytes,
    const slice<data_type, dimensions> &data) const {
    using bits_type = typename Profile::bits_type;
    constexpr static auto side_length = Profile::hypercube_side_length;
    detail::file<Profile> file(data.size());

    size_t stream_pos = file.file_header_length(); // simply skip the header
    file.for_each_superblock([&](auto sb) {
        stream_pos += file.superblock_header_length(); // simply skip the header
        sb.for_each_hypercube([&](auto hc) {
            Profile p;
            bits_type cube[detail::ipow(side_length, Profile::dimensions)] = {};
            stream_pos += p.decode_block(static_cast<const char *>(stream) + stream_pos, cube);
            bits_type *cube_ptr = cube;
            hc.for_each_cell(data, [&](auto *cell) { p.store_value(cell, *cube_ptr++); });
        });
    });
    stream_pos += detail::unpack_border(data, static_cast<const char *>(stream) + stream_pos,
        side_length);
    return stream_pos;
}


namespace hcde {
    extern template class cpu_encoder<fast_profile<float, 2>>;
    extern template class cpu_encoder<fast_profile<float, 3>>;
    extern template class cpu_encoder<strong_profile<float, 2>>;
    extern template class cpu_encoder<strong_profile<float, 3>>;
}
