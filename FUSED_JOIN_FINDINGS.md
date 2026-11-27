# Fused supplier join and grouped aggregation

The CPU experiment targets an inner integer equijoin with a proven unique
build key, an integer group column from the build side, and `COUNT(*)` with
optional `SUM(integer)` from the probe side. PostgreSQL child plans still
apply filters with normal snapshot semantics.

The executor maps each build key directly to a compact group index. It
streams projected probe slots through the lookup and updates count and sum in
place. Its output consists of group slots, so the supplier query on SF5 can
consume 29,999,795 line items and emit 25 rows. The path checks accumulator
overflow and builds under the existing host memory budget. It accounts for
nullable keys, nullable groups and all-NULL sums.

## Measurement protocol

The server is PostgreSQL 18.6 on the eight-vCPU AMD EPYC host used for the
original join experiments. All timings are median PostgreSQL
`EXPLAIN ANALYZE` execution time. Each scale uses a single repeatable-read
snapshot, an untimed result verification pass, one warmup, and three
randomized timed rounds. Reported worker counts are the workers observed in
the plans. Native preaggregation groups line items by supplier before
joining to the supplier table and keeps the same result types and values.

```bash
python3 bench/join_benchmark.py --dbname tpch_sf1 tpch_sf5 \
  --cases supplier_grouped --modes native native_preagg fused_cpu \
  --native-worker-sweep 0 2 4 --trials 3 --warmups 1 --output results/fused-main
```

## Serial fused and native execution

| Plan | Workers launched | SF1 | SF5 |
| --- | ---: | ---: | ---: |
| Native join and aggregation | 0 | 2,288 ms | 12,107 ms |
| Native join and aggregation | 2 | 812 ms | 4,190 ms |
| Native join and aggregation | 4 | 485 ms | 2,500 ms |
| Native fact preaggregation | 0 | 1,730 ms | 9,259 ms |
| Native fact preaggregation | 2 | 617 ms | 3,195 ms |
| Native fact preaggregation | 4 | 373 ms | 1,948 ms |
| Fused CPU | 0 | 1,080 ms | 5,779 ms |

Serial fusion cut elapsed time by 53% at SF1 and 52% at SF5 against serial
native execution. The fused SF5 node spent 15 ms building and about 5,763 ms
scanning the probe child and reducing the rows. The 25-group final output
is small. PostgreSQL's parallel native plan has a substantial advantage
even after the fused operator removes the joined output boundary. Fact
preaggregation with four workers is the strongest native reference in this
experiment.

All result digests matched. The serial path also passed exact SQL multiset
and type comparisons on fixtures covering NULL groups and join keys, all-NULL
sums, signed extremes, missing keys, duplicates on the probe side, empty
builds, rescan and unsupported query shapes.

## Worker-local fusion

The separate `pg_gftr.fused_parallel` opt-in uses PostgreSQL's parallel-aware
sequential fact scan. Each participating backend builds a private dimension
lookup and updates local int8 aggregate transition states. Gather reads those
small states, and PostgreSQL's native final hash aggregate combines them.
EXPLAIN identifies the partial custom scan and labels its counters as local
to a participant. Budget for
one build per participant when setting the operator's host memory limit.

The parallel candidate requires an eligible parallel-safe base plan, a
hashable group key and a top-level query. The serial path handles cases such
as correlated rescans. GPU fusion remains an operator-level research
prototype. Existing measurements show that uploading host-resident input
dominates its kernel time. See [JOIN_FINDINGS.md](JOIN_FINDINGS.md).

| Plan | Workers launched | SF1 | SF5 |
| --- | ---: | ---: | ---: |
| Native join and aggregation | 2 | 785 ms | 4,181 ms |
| Native fact preaggregation | 2 | 596 ms | 3,164 ms |
| Parallel fused CPU | 2 | 356 ms | 1,954 ms |
| Native join and aggregation | 4 | 478 ms | 2,489 ms |
| Native fact preaggregation | 4 | 380 ms | 1,990 ms |
| Parallel fused CPU | 4 | 214 ms | 1,078 ms |

All result digests matched. With four workers, parallel fusion was
2.23× as fast as the original native query at SF1 and 2.31× as fast at
SF5. It was 1.78× and 1.85× as fast as native fact preaggregation.
Gather handled 75 partial group rows with two workers and 125 with four,
including the leader, before reducing them to 25 final groups. Both scales
launched every requested worker in every timed trial.

The planner also selected the four-worker fused path with
`fused_enabled` and `fused_parallel` on and `fused_force` off in the two
TPC-H databases. Fusion stays opt-in. The measurements cover one join and
grouping shape in isolated sessions. The SQL tests verify exact results
with workers, with leader participation disabled, and after falling back
to serial fusion when the worker limit is zero.

```bash
python3 bench/join_benchmark.py --dbname tpch_sf1 tpch_sf5 \
  --cases supplier_grouped --modes native native_preagg fused_parallel \
  --native-worker-sweep 2 4 --fused-worker-sweep 2 4 \
  --trials 3 --warmups 1 --output results/fused-parallel
```
