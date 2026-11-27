#include "gftr_join.h"
#include "gftr_join_lookup.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

void check(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

template <typename T>
T *allocate_device(size_t count)
{
    T *pointer = nullptr;
    check(cudaMalloc(&pointer, count * sizeof(T)), "allocate device join memory");
    return pointer;
}

constexpr int threads = 256;

int blocks(int32_t count)
{
    return static_cast<int>(std::min<int64_t>(
        ((int64_t) count + threads - 1) / threads, 65535));
}

template <bool Dense>
__global__ void build_lookup(void *table, GftrJoinLayout layout,
                             const int32_t *keys, int32_t count, int *duplicate)
{
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (int64_t) blockDim.x * gridDim.x) {
        int32_t key = keys[i];
        if constexpr (Dense) {
            int32_t *indices = static_cast<int32_t *>(table);
            size_t slot = (size_t) ((int64_t) key - layout.minimum);
            if (atomicCAS(&indices[slot], -1, (int32_t) i) != -1)
                atomicExch(duplicate, 1);
        } else {
            GftrJoinEntry *entries = static_cast<GftrJoinEntry *>(table);
            size_t mask = layout.slots - 1;
            size_t slot = gftr_join_hash(key, mask);
            GftrJoinEntry entry = gftr_join_entry(key, (int32_t) i);
            for (;;) {
                GftrJoinEntry previous = atomicCAS(&entries[slot], GFTR_JOIN_EMPTY, entry);
                if (previous == GFTR_JOIN_EMPTY)
                    break;
                if (gftr_join_is_key(previous, key)) {
                    atomicExch(duplicate, 1);
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
    }
}

template <bool Dense>
__global__ void probe_lookup(const void *table, GftrJoinLayout layout,
                             const int32_t *keys, const uint8_t *valid,
                             int32_t count, int32_t *matches)
{
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += (int64_t) blockDim.x * gridDim.x) {
        int32_t match = -1;
        if (!valid || valid[i]) {
            if constexpr (Dense)
                match = gftr_join_dense_lookup(static_cast<const int32_t *>(table),
                                               layout, keys[i]);
            else
                match = gftr_join_hash_lookup(static_cast<const GftrJoinEntry *>(table),
                                              layout.slots, keys[i]);
        }
        matches[i] = match;
    }
}

struct CudaJoinBuild {
    int device;
    int32_t batch_rows;
    GftrJoinLayout layout;
    cudaStream_t stream = nullptr;
    cudaEvent_t events[4]{};
    uint8_t *table = nullptr;
    int32_t *build_keys = nullptr;
    int *duplicate = nullptr;
    int has_duplicate = 0;
    int32_t *probe_keys = nullptr;
    uint8_t *probe_valid = nullptr;
    int32_t *matches = nullptr;

    CudaJoinBuild(int device, int32_t batch_rows, GftrJoinLayout layout) :
        device(device), batch_rows(batch_rows), layout(layout) {}

    ~CudaJoinBuild()
    {
        if (!stream)
            return;
        cudaSetDevice(device);
        cudaStreamSynchronize(stream);
        if (table)
            cudaFree(table);
        if (build_keys)
            cudaFree(build_keys);
        if (duplicate)
            cudaFree(duplicate);
        if (probe_keys)
            cudaFree(probe_keys);
        if (probe_valid)
            cudaFree(probe_valid);
        if (matches)
            cudaFree(matches);
        for (auto event : events) {
            if (event)
                cudaEventDestroy(event);
        }
        cudaStreamDestroy(stream);
    }

    void record(int event)
    {
        check(cudaEventRecord(events[event], stream), "record join event");
    }

    double time_ms(int first, int last)
    {
        float ms;
        check(cudaEventElapsedTime(&ms, events[first], events[last]),
              "measure join event");
        return ms;
    }

    void finish()
    {
        record(3);
        check(cudaEventSynchronize(events[3]), "synchronize CUDA join batch");
    }

    void initialize(const int32_t *keys, int32_t count, size_t budget,
                    GftrJoinTiming &timing)
    {
        size_t free_bytes, total_bytes;
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "get available device join memory");
        size_t limit = std::min(budget, free_bytes - free_bytes / 10);
        size_t scratch_bytes = (size_t) count * sizeof(int32_t) + sizeof(int);
        size_t batch_bytes = (size_t) batch_rows * (2 * sizeof(int32_t) + sizeof(uint8_t));
        if (layout.bytes > limit || std::max(scratch_bytes, batch_bytes) > limit - layout.bytes)
            throw std::runtime_error("CUDA join lookup and batch buffers exceed the device memory budget");

        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create join stream");
        for (auto &event : events)
            check(cudaEventCreate(&event), "create join event");

        table = allocate_device<uint8_t>(layout.bytes);
        build_keys = allocate_device<int32_t>(count);
        duplicate = allocate_device<int>(1);
        const size_t key_bytes = (size_t) count * sizeof(int32_t);
        record(0);
        check(cudaMemcpyAsync(build_keys, keys, key_bytes, cudaMemcpyHostToDevice, stream),
              "copy join build keys to device");
        record(1);
        check(cudaMemsetAsync(table, 0xff, layout.bytes, stream), "initialize join lookup");
        check(cudaMemsetAsync(duplicate, 0, sizeof(int), stream), "initialize join duplicate flag");
        if (layout.dense)
            build_lookup<true><<<blocks(count), threads, 0, stream>>>(
                table, layout, build_keys, count, duplicate);
        else
            build_lookup<false><<<blocks(count), threads, 0, stream>>>(
                table, layout, build_keys, count, duplicate);
        check(cudaGetLastError(), "launch join build kernel");
        record(2);
        check(cudaMemcpyAsync(&has_duplicate, duplicate, sizeof(int), cudaMemcpyDeviceToHost,
                              stream), "read join duplicate flag");
        finish();
        if (has_duplicate)
            throw std::runtime_error("duplicate key on the unique join build side");
        timing.h2d_bytes = key_bytes;
        timing.d2h_bytes = sizeof(int);
        timing.h2d_ms = time_ms(0, 1);
        timing.d2h_ms = time_ms(2, 3);

        check(cudaFree(build_keys), "free join build key buffer");
        build_keys = nullptr;
        check(cudaFree(duplicate), "free join duplicate flag");
        duplicate = nullptr;
        probe_keys = allocate_device<int32_t>(batch_rows);
        probe_valid = allocate_device<uint8_t>(batch_rows);
        matches = allocate_device<int32_t>(batch_rows);
    }

    void probe(const int32_t *keys, const uint8_t *valid, int32_t count,
               int32_t *output, GftrJoinTiming &timing)
    {
        check(cudaSetDevice(device), "select CUDA join device");
        size_t key_bytes = (size_t) count * sizeof(int32_t);
        size_t valid_bytes = valid ? (size_t) count * sizeof(uint8_t) : 0;
        record(0);
        check(cudaMemcpyAsync(probe_keys, keys, key_bytes, cudaMemcpyHostToDevice, stream),
              "copy join probe keys to device");
        if (valid)
            check(cudaMemcpyAsync(probe_valid, valid, valid_bytes, cudaMemcpyHostToDevice,
                                  stream), "copy join probe validity to device");
        record(1);
        const uint8_t *valid_on_device = valid ? probe_valid : nullptr;
        if (layout.dense)
            probe_lookup<true><<<blocks(count), threads, 0, stream>>>(
                table, layout, probe_keys, valid_on_device, count, matches);
        else
            probe_lookup<false><<<blocks(count), threads, 0, stream>>>(
                table, layout, probe_keys, valid_on_device, count, matches);
        check(cudaGetLastError(), "launch join probe kernel");
        record(2);
        size_t output_bytes = (size_t) count * sizeof(int32_t);
        check(cudaMemcpyAsync(output, matches, output_bytes, cudaMemcpyDeviceToHost, stream),
              "copy join matches to host");
        finish();

        timing.h2d_bytes += key_bytes + valid_bytes;
        timing.d2h_bytes += output_bytes;
        timing.h2d_ms += time_ms(0, 1);
        timing.probe_ms += time_ms(1, 2);
        timing.d2h_ms += time_ms(2, 3);
    }
};

} // namespace

extern "C" int
gftr_join_cuda_build(const int32_t *keys, int32_t count, int device,
                     int32_t batch_rows, size_t budget,
                     GftrJoinBuild **out, GftrJoinTiming *timing,
                     char error[GFTR_ERROR_SIZE])
{
    *out = nullptr;
    *timing = GftrJoinTiming{};
    try {
        auto start = Clock::now();
        if (count < 0 || (count > 0 && !keys) || batch_rows <= 0)
            throw std::runtime_error("invalid CUDA join build input");
        check(cudaSetDevice(device), "select CUDA join device");
        GftrJoinLayout layout{};
        if (gftr_join_layout(keys, count, &layout))
            throw std::runtime_error("CUDA join lookup size overflow");
        auto build = std::make_unique<CudaJoinBuild>(device, batch_rows, layout);
        if (count > 0)
            build->initialize(keys, count, budget, *timing);
        auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start);
        timing->build_ms = std::max(0.0, elapsed.count() - timing->h2d_ms - timing->d2h_ms);
        timing->build_bytes = layout.bytes;
        timing->algorithm = count == 0 ? "empty" : layout.dense ? "dense" : "hash";
        *out = reinterpret_cast<GftrJoinBuild *>(build.release());
        return 0;
    } catch (const std::exception &ex) {
        std::snprintf(error, GFTR_ERROR_SIZE, "%s", ex.what());
        return -1;
    }
}

extern "C" int
gftr_join_cuda_probe(GftrJoinBuild *opaque, const int32_t *keys,
                     const uint8_t *valid, int32_t count, int32_t *matches,
                     GftrJoinTiming *timing, char error[GFTR_ERROR_SIZE])
{
    auto build = reinterpret_cast<CudaJoinBuild *>(opaque);
    try {
        if (!build || count < 0 || count > build->batch_rows ||
            (count > 0 && (!keys || !matches)))
            throw std::runtime_error("invalid CUDA join probe input or batch size");
        if (count == 0)
            return 0;
        if (build->layout.slots == 0)
            std::fill_n(matches, count, -1);
        else
            build->probe(keys, valid, count, matches, *timing);
        return 0;
    } catch (const std::exception &ex) {
        if (build && build->stream)
            cudaStreamSynchronize(build->stream);
        std::snprintf(error, GFTR_ERROR_SIZE, "%s", ex.what());
        return -1;
    }
}

extern "C" void
gftr_join_cuda_free(GftrJoinBuild *build)
{
    delete reinterpret_cast<CudaJoinBuild *>(build);
}
