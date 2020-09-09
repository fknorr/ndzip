#include <assert.h>
#include <limits.h>
#include <algorithm>
#include <type_traits>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <SYCL/sycl.hpp>

#define NOINLINE

inline uint32_t load(const void *addr) {
    uint32_t v;
    __builtin_memcpy(&v, addr, sizeof v);
    return v;
}

inline void store(void *addr, uint32_t v) {
    __builtin_memcpy(addr, &v, sizeof v);
}

inline uint32_t shl(uint32_t v, unsigned s) {
    return s < 32 ? v << s : 0;
}

inline uint32_t shr(uint32_t v, unsigned s) {
    return s < 32 ? v >> s : 0;
}

inline uint32_t shift(uint32_t v, int s) {
    return shl(v, s) | shr(v, -s);
}


template<typename T>
void xt_step(T *x, size_t n, size_t s) {
    T a, b;
    b = x[0*s];
    for (size_t i = 1; i < n; ++i) {
        a = b;
        b = x[i*s];
        x[i*s] = a ^ b;
    }
}

template<typename T, unsigned Dim> class xt3d_kernel;

template<typename T>
void xt3d(sycl::queue &q, sycl::buffer<T, 1> &buf, size_t n, size_t m, size_t l) {
    q.submit([&](sycl::handler &cgh) {
        auto acc = buf.template get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<xt3d_kernel<T, 1>>(sycl::range<1>{n*m}, [acc, n, m, l](sycl::item<1> item) {
            auto x = static_cast<uint32_t*>(acc.get_pointer());
            auto i = l*item[0];
            xt_step(x+i, l, 1);
        });
    });
    q.submit([&](sycl::handler &cgh) {
        auto acc = buf.template get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<xt3d_kernel<T, 0>>(sycl::range<2>{n, l}, [acc, n, m, l](sycl::item<2> item) {
            auto x = static_cast<uint32_t*>(acc.get_pointer());
            auto i = m*l*item[0];
            auto j = item[1];
            xt_step(x+i+j, m, l);
        });
    });
    q.submit([&](sycl::handler &cgh) {
        auto acc = buf.template get_access<sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<xt3d_kernel<T, 2>>(sycl::range<1>{m*l}, [acc, n, m, l](sycl::item<1> item) {
            auto x = static_cast<uint32_t*>(acc.get_pointer());
            auto i = item[0];
            xt_step(x+i, n, m*l);
        });
    });
}


NOINLINE
size_t writev(const uint32_t *vs, char *out0) {
    auto out = out0;
    for (unsigned i = 0; i < 64; i += 2) {
        unsigned length_codes[2];
        for (unsigned j = 0; j < 2; ++j) {
            if (vs[i+j]) {
                auto lz = __builtin_clz(vs[i+j]);
                auto code = lz >= 14 ? 1 : 15 - lz;
                length_codes[j] = code;
            } else {
                length_codes[j] = 0;
            }
        }
        *out++ = (length_codes[0] << 4) | length_codes[1];
    }

    store(out, 0);
    unsigned bit_pos = 0;

    for (unsigned i = 0; i < 64; i += 1) {
        if (vs[i]) {
            auto lz = __builtin_clz(vs[i]) ;
            auto n_bits = lz >= 14 ? 18 : 31 - lz;

            uint32_t u[2] = {};
            __builtin_memcpy(u, out, 4);
            uint64_t v;
            __builtin_memcpy(&v, &u, sizeof v);
            auto bits = vs[i] & ~(~uint64_t{} << n_bits);
            v |= __builtin_bswap64(uint64_t(bits) << (64u - bit_pos - n_bits));
            __builtin_memcpy(out, &v, 8);

            bit_pos += n_bits;
            out += bit_pos / 32 * 4;
            bit_pos %= 32;
        }
    }

    return out-out0+(bit_pos+7)/8;
}


void die(const char *what) {
    perror(what);
    exit(1);
}


int main(int argc, char **argv) {
    auto fn_in = argv[1];
    auto fn_out = argv[2];
    auto n = (unsigned)atoi(argv[3]);
    auto m = (unsigned)atoi(argv[4]);
    auto l = (unsigned)atoi(argv[5]);

    auto fd_in = open(fn_in, O_RDONLY);
    if (fd_in == -1) die("open");

    struct stat stat_in;
    if (fstat(fd_in, &stat_in) == -1) die("stat");
    auto in_size = static_cast<size_t>(stat_in.st_size);

    auto mem_in = static_cast<uint32_t*>(mmap(NULL, in_size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd_in, 0));
    if (mem_in == MAP_FAILED) die("mmap");

    auto fd_out = open(fn_out, O_RDWR|O_CREAT, 0666);
    if (fd_out == -1) die("open");

    size_t out_maxsize = n*m*l*5;
    if (ftruncate(fd_out, out_maxsize) == -1) die("ftruncate");

    auto mem_out = static_cast<char*>(mmap(NULL, out_maxsize, PROT_READ|PROT_WRITE, MAP_SHARED, fd_out, 0));
    if (mem_out == MAP_FAILED) die("mmap");

    sycl::buffer<uint32_t, 1> buf_in(mem_in, n*m*l);

    {
        sycl::queue q;
        xt3d(q, buf_in, n, m, l);

        sycl::buffer<char, 1> buf_out(mem_out, out_maxsize);

        q.submit([&](sycl::handler &cgh) {
            auto acc_out = buf_out.get_access<sycl::access::mode::discard_write>(cgh);
            cgh.fill(acc_out, char{});
        });
        q.submit([&](sycl::handler &cgh) {
            auto acc_in = buf_in.get_access<sycl::access::mode::read>(cgh);
            auto acc_out = buf_out.get_access<sycl::access::mode::read_write>(cgh);
            auto acc_temp = sycl::accessor<char, 1, sycl::access::mode::read_write,
                    sycl::access::target::local>(64*5*32, cgh);
            cgh.parallel_for<class write>(
                    sycl::nd_range<1>{sycl::range<1>{((n*m*l+63)/64+31)/32*32}, sycl::range<1>{32}},
                    [=](sycl::nd_item<1> nd_item) {
                auto start_index = nd_item.get_global_id(0) * 64;
                if (start_index < n*m*l) {
                    auto o = static_cast<char*>(acc_temp.get_pointer()) + nd_item.get_local_id(0)*64*5;
                    writev(static_cast<const uint32_t*>(acc_in.get_pointer()) + start_index, o);
                    memcpy(static_cast<char*>(acc_out.get_pointer()) + start_index*5, o, 64*5);
                }
            });
        });
        q.submit([&](sycl::handler &cgh) {
            auto buf_acc = buf_in.get_access<sycl::access::mode::read_write>(cgh);
            cgh.update_host(buf_acc);
        });
        q.wait();
    }

    auto out_size = out_maxsize;//o-static_cast<char*>(mem_out);

    if (munmap(mem_out, out_maxsize) == -1) die("munmap");
    if (ftruncate(fd_out, out_size) == -1) die("ftruncate");
    if (close(fd_out) == -1) die("close");
    if (munmap(mem_in, in_size) == -1) die("munmap");
    if (close(fd_in) == -1) die("close");

    printf("%zu => %zu (ratio %.4g)\n", in_size, out_size, double(in_size)/out_size);
}

