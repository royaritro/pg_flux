# GPU join findings

The `gpu-joins` branch adds an experimental PostgreSQL 18 custom join node.
It supports a read-only inner equijoin on one `integer` key between base
tables when the build key is proven unique. The extension uses a bounded
probe batch and a compact dense lookup or an open-addressed hash table.
Join acceleration is opt-in.

## TPC-H workload and measurements

The host is PostgreSQL 18.6 on eight AMD EPYC vCPUs with an NVIDIA A100
SXM4 80 GB and CUDA 12.6. The generated SF1 and SF5 datasets contain
6,001,215 and 29,999,795 line items. The cases are TPC-H-derived SQL
microbenchmarks. The native baseline uses the server's default planner
settings, including up to two workers per Gather and 4 MiB `work_mem`.
The tuned native run uses 256 MiB `work_mem`.

```bash
python3 bench/join_benchmark.py --dbname tpch_sf1 tpch_sf5 \
  --modes native native_tuned cpu cuda auto native_gpu_agg \
  --trials 3 --warmups 1 --output results/join-main
```

The runner checks result digests and physical plan providers before reporting
median `EXPLAIN ANALYZE` execution times. Each database uses one session and a
repeatable-read snapshot. One randomized warmup round precedes three
randomized measured rounds. The timing includes server execution and
accelerator transfers. Query result delivery to the client is outside the
measurement. The CPU and CUDA custom join variants are forced only for
comparison. Auto mode offers the CUDA path at its provisional planner cost.

| SF1 case | Native | Native tuned | Compact CPU | CUDA | Auto |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supplier, grouped | 804.0 ms | 806.1 ms | 2,163.0 ms | 2,287.1 ms | 792.8 ms |
| Part, payloads | 897.1 ms | 877.0 ms | 1,921.3 ms | 1,850.7 ms | 910.4 ms |
| Orders, broad | 1,081.0 ms | 942.5 ms | 1,243.9 ms | 1,364.1 ms | 1,050.1 ms |
| Orders, selective | 171.3 ms | 182.6 ms | 1,194.0 ms | 1,256.5 ms | 175.6 ms |
| Supplier, bounded projection | 40.4 ms | 39.2 ms | 19.7 ms | 29.9 ms | 37.6 ms |

| SF5 case | Native | Native tuned | Compact CPU | CUDA | Auto |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supplier, grouped | 4,151.0 ms | 4,171.9 ms | 10,956.6 ms | 10,886.9 ms | 4,163.3 ms |
| Part, payloads | 5,106.5 ms | 8,380.4 ms | 10,269.9 ms | 10,169.7 ms | 5,081.0 ms |
| Orders, broad | 5,074.6 ms | 5,151.7 ms | 6,464.7 ms | 6,740.8 ms | 5,095.7 ms |
| Orders, selective | 2,624.9 ms | 2,587.2 ms | 6,613.3 ms | 5,923.7 ms | 2,680.4 ms |
| Supplier, bounded projection | 38.2 ms | 34.0 ms | 26.3 ms | 42.2 ms | 38.3 ms |

All modes returned matching digests at both scales. The bounded supplier
projection returns 40,278 rows. Compact CPU was 1.99× and 1.29× as fast as
the faster native mode at SF1 and SF5, respectively. Forced CUDA improved
the same query at SF1 by 24% and lost at SF5. Native execution won every
broad and selective case. Auto chose native plans for all five workloads at
both scales. These measurements do not support enabling acceleration by
default or calibrating a broad GPU selection rule.

The existing CUDA aggregation over a native join took 1,693.9 ms at SF1 and
9,502.2 ms at SF5 on the grouped supplier query. The SF5 PostgreSQL CUDA
join spent median 5,022 ms scanning and packing probe rows, approximately
1,018 ms constructing output slots, and 3.9 ms in the probe kernel. It
returned 119,999,184 bytes of match data to the host. The equivalent native
plan could combine a parallel join with partial aggregation, which limited
the number of rows it needed to return from the workers. The custom join
feeds one PostgreSQL output tuple per match to the upper plan. Cutting
per-tuple work is the priority for broad joins.

## Fused supplier prototype

The standalone prototype uses the TPC-H supplier-to-nation mapping and
returns the 25 groups for `COUNT(*)` and `SUM(l_linenumber)`. A GPU kernel
looks up each supplier and accumulates into per-warp histograms. The dimension
stays on the GPU across trials. Every trial matched the compact CPU result,
and both scales matched native SQL output byte for byte.

| Scale | Compact CPU | Fused CUDA | Fact H2D | GPU kernel | Result D2H |
| --- | ---: | ---: | ---: | ---: | ---: |
| SF1 | 9.7 ms | 62.4 ms | 59.8 ms | 0.075 ms | 2.462 ms |
| SF5 | 53.2 ms | 153.6 ms | 152.1 ms | 0.337 ms | 0.023 ms |

These are seven warm operator medians using already parsed host arrays.
File reading and decoding took about 584 ms at SF1 and 3,461 ms at SF5.
Initial GPU setup took about 314 ms and 397 ms, respectively. The CUDA
operator uploads about 48 MB and 240 MB of fact columns before returning
400 bytes at either scale. Fusion removes the large match transfer and
PostgreSQL output slots from this experiment. The compact CPU operation
still wins when the facts start on the host. A PostgreSQL fusion path is
future work, with parallel packing or an already device-resident input as
the most useful next comparison.

```bash
nvcc -O3 -std=c++17 -gencode arch=compute_80,code=sm_80 bench/fused_supplier.cu   -o /tmp/fused-supplier
/tmp/fused-supplier /home/ubuntu/tpch/data/sf1 7 >sf1-groups.txt 2>sf1-fused.csv
/tmp/fused-supplier /home/ubuntu/tpch/data/sf5 7 >sf5-groups.txt 2>sf5-fused.csv
```

## Cache and transfer experiment

The host exposes 512 KiB L2 per vCPU and 16 MiB shared L3. The A100 reports
40 MiB L2 with a maximum 25 MiB persisting set-aside. A 32-bit dense row-index
table is about 195 KiB for the SF5 supplier key range, 3.8 MiB for part, and
114 MiB for orders. Payload buffers and database tuple overhead are separate.

`bench/join_transfer.cu` measured 8,388,608 synthetic probes per trial at those
supplier and orders range sizes. The build table was already resident. Seven
measured trials followed one warmup. Each CUDA trial sent 40 MiB and returned
32 MiB. Values below are median wall milliseconds, including the probe
routine's transfers and synchronization.

| Lookup and key order | Compact CPU | CUDA pageable | CUDA pinned |
| --- | ---: | ---: | ---: |
| 195 KiB, sequential | 9.31 | 94.82 | 90.46 |
| 195 KiB, randomized | 11.37 | 94.79 | 90.79 |
| 114 MiB, sequential | 14.50 | 90.42 | 86.41 |
| 114 MiB, randomized | 130.72 | 90.46 | 76.89 |

CUDA probe kernels took 0.51–0.96 ms across a trial. The difference between
the large sequential and randomized CPU cases makes probe locality a planning
input. The timing includes neither PostgreSQL tuple access nor build time, so
the randomized GPU win is a backend-level result. Pinned staging is most
promising for the large randomized case. L2 persistence and overlapping DMA
were outside this measurement.

Reproduce the transfer test after setting `CUDA_HOME` and adding its `bin`
directory to `PATH`.

```bash
cc -std=c11 -D_POSIX_C_SOURCE=200809L -O2 -Isrc -c src/gftr_join_cpu.c -o /tmp/join-cpu.o
nvcc -O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -Isrc \
  bench/join_transfer.cu src/gftr_join_cuda.cu /tmp/join-cpu.o -o /tmp/join-transfer
/tmp/join-transfer
```

## Follow-up experiments

- Compare the compact CPU lookup against native index joins and parallel hash
  joins before adding accelerator setup cost. For 195 KiB builds and
  locality-friendly streams the CPU probe has substantial headroom.
- Measure packing and materialization before changing the CUDA lookup kernel.
  Narrow projection and consuming GPU output on the device can save both
  host work and PCIe traffic.
- Compare plain caching with an L2 access policy window only for reusable,
  repeatedly accessed build data. For large randomized lookups, test radix
  partitioning against direct addressing on CPU and CUDA, counting the
  partition pass and output reconstruction in the result.
- Measure end-to-end latency across selectivity and batch size before enabling
  automatic selection. Include time-to-first-row for queries that can stop
  early, and run concurrency separately from isolated measurements.

[CUDA data movement and coalescing](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-best-practices-guide/index.html) ·
[CUDA L2 persistence](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#l2-cache-set-aside-for-persisting-accesses) ·
[GFTR research](https://arxiv.org/abs/2312.00720)
