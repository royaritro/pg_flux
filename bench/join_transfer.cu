/*
 * Isolate probe time and transfer time with TPC-H SF5 supplier and orders
 * key ranges. Inputs are synthetic, with equal match rates across key orders.
 * Build and input generation are outside the measured probe trials.
 */
#include "gftr_join.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int32_t probe_rows = 1 << 23;
constexpr int32_t batch_rows = 1 << 18;
constexpr size_t budget = size_t{512} << 20;
constexpr int trials = 7;

void check_cuda(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

void check_join(int status, const char *error)
{
    if (status != 0)
        throw std::runtime_error(error);
}

template <typename T>
struct HostBuffer {
    T *data = nullptr;
    bool pinned;

    explicit HostBuffer(bool pinned) : pinned(pinned)
    {
        const size_t bytes = probe_rows * sizeof(T);
        if (pinned)
            check_cuda(cudaMallocHost(&data, bytes));
        else {
            data = static_cast<T *>(std::malloc(bytes));
            if (!data)
                throw std::bad_alloc();
        }
    }

    ~HostBuffer()
    {
        if (pinned)
            cudaFreeHost(data);
        else
            std::free(data);
    }

    HostBuffer(const HostBuffer &) = delete;
    HostBuffer &operator=(const HostBuffer &) = delete;
};

struct Scenario {
    const char *name;
    int32_t build_rows;
    int32_t key_step;
};

void run_trials(const Scenario &scenario, GftrJoinBuild *build,
                const GftrJoinTiming &build_timing, bool gpu, bool pinned,
                bool random_order)
{
    HostBuffer<int32_t> keys(pinned);
    HostBuffer<uint8_t> valid(pinned);
    HostBuffer<int32_t> matches(pinned);
    uint32_t random = 301;

    for (int32_t i = 0; i < probe_rows; i++) {
        random = random * UINT32_C(1664525) + UINT32_C(1013904223);
        int32_t row = (random_order ? random : (uint32_t) i) % scenario.build_rows;
        keys.data[i] = 1 + row * scenario.key_step;
        valid.data[i] = i % 17 != 0;
    }

    char error[GFTR_ERROR_SIZE]{};
    for (int trial = 0; trial <= trials; trial++) {
        GftrJoinTiming timing{};
        auto start = std::chrono::steady_clock::now();
        for (int32_t offset = 0; offset < probe_rows; offset += batch_rows) {
            int32_t count = std::min(batch_rows, probe_rows - offset);
            if (gpu)
                check_join(gftr_join_cuda_probe(build, keys.data + offset, valid.data + offset,
                                                count, matches.data + offset, &timing, error), error);
            else
                check_join(gftr_join_cpu_probe(build, keys.data + offset, valid.data + offset,
                                               count, matches.data + offset, &timing, error), error);
        }
        double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

        for (int32_t i = 0; i < probe_rows; i++) {
            int32_t expected = valid.data[i] ? (keys.data[i] - 1) / scenario.key_step : -1;
            if (matches.data[i] != expected)
                throw std::runtime_error("probe produced an incorrect match");
        }

        if (trial > 0)
            std::printf("%s,%s,%zu,%s,%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%zu,%zu\n",
                        scenario.name, build_timing.algorithm, build_timing.build_bytes,
                        gpu ? "cuda" : "cpu", pinned ? "pinned" : "pageable",
                        random_order ? "random" : "sequential", trial,
                        probe_rows, batch_rows, wall_ms, timing.h2d_ms, timing.probe_ms,
                        timing.d2h_ms, timing.h2d_bytes, timing.d2h_bytes);
    }
}

} // namespace

int main()
{
    try {
        check_cuda(cudaSetDevice(0));
        std::puts("scenario,algorithm,build_bytes,backend,host_memory,key_order,trial,"
                  "probe_rows,batch_rows,wall_ms,h2d_ms,probe_ms,d2h_ms,h2d_bytes,d2h_bytes");
        for (Scenario scenario : {Scenario{"supplier", 50000, 1}, Scenario{"orders", 7500000, 4}}) {
            std::vector<int32_t> build_keys(scenario.build_rows);
            for (int32_t i = 0; i < scenario.build_rows; i++)
                build_keys[i] = 1 + i * scenario.key_step;

            GftrJoinBuild *build = nullptr;
            GftrJoinTiming timing{};
            char error[GFTR_ERROR_SIZE]{};
            check_join(gftr_join_cpu_build(build_keys.data(), scenario.build_rows, budget,
                                          &build, &timing, error), error);
            {
                std::unique_ptr<GftrJoinBuild, decltype(&gftr_join_cpu_free)>
                    owner(build, gftr_join_cpu_free);
                run_trials(scenario, build, timing, false, false, false);
                run_trials(scenario, build, timing, false, false, true);
            }

            check_join(gftr_join_cuda_build(build_keys.data(), scenario.build_rows, 0, batch_rows,
                                           budget, &build, &timing, error), error);
            std::unique_ptr<GftrJoinBuild, decltype(&gftr_join_cuda_free)>
                owner(build, gftr_join_cuda_free);
            for (bool random_order : {false, true})
                for (bool pinned : {false, true})
                    run_trials(scenario, build, timing, true, pinned, random_order);
        }
        return 0;
    } catch (const std::exception &ex) {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
