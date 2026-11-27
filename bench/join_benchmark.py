#!/usr/bin/env python3
"""Benchmark native and custom join paths on loaded PostgreSQL TPC-H data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import subprocess

QUERIES = {
    "supplier_grouped": """
SELECT s.s_nationkey, count(*), sum(l.l_linenumber)
FROM lineitem l JOIN supplier s ON l.l_suppkey = s.s_suppkey
GROUP BY s.s_nationkey
ORDER BY s.s_nationkey""",
    "part_payload": """
SELECT count(*), sum(p.p_size), sum(p.p_partkey), sum(l.l_linenumber)
FROM lineitem l JOIN part p ON l.l_partkey = p.p_partkey""",
    "orders_broad": """
SELECT count(*), sum(o.o_custkey)
FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey""",
    "orders_selective": """
SELECT count(*), sum(l.l_partkey)
FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey
WHERE o.o_orderdate >= DATE '1995-01-01'
  AND o.o_orderdate < DATE '1995-02-01'""",
    "supplier_projected": """
SELECT l.l_orderkey, l.l_linenumber, l.l_suppkey, s.s_nationkey
FROM lineitem l JOIN supplier s ON l.l_suppkey = s.s_suppkey
WHERE l.l_orderkey BETWEEN 1 AND 40000
ORDER BY l.l_orderkey, l.l_linenumber""",
}
PREAGG_SUPPLIER_QUERY = """
SELECT s.s_nationkey,
       sum(l.line_count)::bigint AS count,
       sum(l.line_number_sum)::bigint AS sum
FROM (
    SELECT l_suppkey, count(*) AS line_count, sum(l_linenumber) AS line_number_sum
    FROM lineitem
    GROUP BY l_suppkey
) l JOIN supplier s ON l.l_suppkey = s.s_suppkey
GROUP BY s.s_nationkey
ORDER BY s.s_nationkey"""
MODES = ("native", "native_tuned", "cpu", "cuda", "native_gpu_agg", "auto",
         "fused_cpu", "native_preagg", "fused_parallel")
FUSED_MODES = ("fused_cpu", "fused_parallel")
CUSTOM_JOIN_MODES = ("cpu", "cuda", "auto") + FUSED_MODES
GROUPED_MODES = ("native_gpu_agg", "native_preagg") + FUSED_MODES
NATIVE_MODES = ("native", "native_tuned", "native_preagg")
RECORD = re.compile(r"^GFTR_JOIN_RECORD (\w+)$", re.MULTILINE)
SETTINGS = (
    "server_version",
    "work_mem",
    "hash_mem_multiplier",
    "max_parallel_workers",
    "max_parallel_workers_per_gather",
    "jit",
    "shared_buffers",
    "effective_cache_size",
    "track_io_timing",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psql", default="psql")
    parser.add_argument("--dbname", nargs="+", default=["tpch_sf1", "tpch_sf5"],
                        help="database names or libpq connection strings, in run order")
    parser.add_argument("--cases", choices=QUERIES, nargs="+", default=list(QUERIES))
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES[:4]))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=714)
    parser.add_argument("--statement-timeout-ms", type=int, default=300_000)
    parallelism = parser.add_mutually_exclusive_group()
    parallelism.add_argument("--serial", action="store_true", help="disable plan parallelism")
    parallelism.add_argument("--native-worker-sweep", nargs="+", type=int,
                             metavar="N", help="benchmark each native mode with each limit "
                             "for max_parallel_workers_per_gather, e.g. 0 2 4")
    parser.add_argument("--fused-worker-sweep", nargs="+", type=int, metavar="N",
                        help="benchmark fused_parallel with each positive limit for "
                        "max_parallel_workers_per_gather, e.g. 2 4")
    parser.add_argument("--disable-jit", action="store_true")
    parser.add_argument("--tuned-work-mem-mb", type=int, default=256)
    parser.add_argument("--join-batch-rows", type=int, default=131_072)
    parser.add_argument("--join-max-host-mb", type=int, default=256)
    parser.add_argument("--join-max-device-mb", type=int, default=4096)
    parser.add_argument("--agg-max-input-mb", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/join"))
    args = parser.parse_args()
    if (args.trials < 1 or args.warmups < 0 or args.device < 0
            or min(args.tuned_work_mem_mb, args.join_batch_rows, args.join_max_host_mb,
                   args.join_max_device_mb, args.agg_max_input_mb,
                   args.statement_timeout_ms) < 1):
        parser.error("invalid trial count, device, memory budget, batch size or timeout")
    if len(set(args.modes)) != len(args.modes) or len(set(args.cases)) != len(args.cases):
        parser.error("each mode and case may appear only once")
    if args.native_worker_sweep is not None:
        workers = args.native_worker_sweep
        if min(workers) < 0 or len(set(workers)) != len(workers):
            parser.error("--native-worker-sweep requires distinct nonnegative worker counts")
        if not set(args.modes).intersection(NATIVE_MODES):
            parser.error("--native-worker-sweep requires a native mode")
    if "fused_parallel" in args.modes and args.serial:
        parser.error("fused_parallel requires parallel workers")
    if args.fused_worker_sweep is not None:
        workers = args.fused_worker_sweep
        if min(workers) < 1 or len(set(workers)) != len(workers):
            parser.error("--fused-worker-sweep requires distinct positive worker counts")
        if "fused_parallel" not in args.modes or "supplier_grouped" not in args.cases:
            parser.error("--fused-worker-sweep requires fused_parallel and supplier_grouped")
    if not any(case_modes(args, case) for case in args.cases):
        parser.error(f"{', '.join(GROUPED_MODES)} apply to supplier_grouped only")
    return args


def case_modes(args, case):
    return [mode for mode in args.modes
            if mode not in GROUPED_MODES or case == "supplier_grouped"]


def mode_variants(args, case):
    variants = {}
    for mode in case_modes(args, case):
        worker_limits = None
        if mode in NATIVE_MODES:
            worker_limits = args.native_worker_sweep
        elif mode == "fused_parallel":
            worker_limits = args.fused_worker_sweep
        if worker_limits is None:
            variants[mode] = (mode, None)
        else:
            for workers in worker_limits:
                variants[f"{mode}_w{workers}"] = (mode, workers)
    return variants


def query_for_mode(case, mode):
    # The rewrite combines partial aggregates over each supplier key. Both
    # outer casts preserve the original COUNT(*) and SUM(integer) types.
    return PREAGG_SUPPLIER_QUERY if mode == "native_preagg" else QUERIES[case]


def configure(args, mode, workers=None):
    settings = [
        f"SET work_mem='{args.tuned_work_mem_mb}MB';"
        if mode == "native_tuned" else "RESET work_mem;",
        f"SET pg_gftr.enabled={'on' if mode == 'native_gpu_agg' else 'off'};",
    ]
    if workers is not None:
        settings.append(f"SET max_parallel_workers_per_gather={workers};")
    elif args.native_worker_sweep is not None or args.fused_worker_sweep is not None:
        settings.append("RESET max_parallel_workers_per_gather;")
    if mode in FUSED_MODES:
        settings += [
            "SET pg_gftr.join_enabled=off;",
            "SET pg_gftr.join_force=off;",
            "SET pg_gftr.join_backend='cpu';",
        ]
    elif mode in CUSTOM_JOIN_MODES:
        backend = "cpu" if mode == "cpu" else "cuda"
        settings += [
            "SET pg_gftr.join_enabled=on;",
            f"SET pg_gftr.join_force={'off' if mode == 'auto' else 'on'};",
            f"SET pg_gftr.join_backend='{backend}';",
        ]
    else:
        # Also run native and aggregate-only modes against pre-join builds.
        settings.append(
            "SELECT 'SET pg_gftr.join_enabled=off;' "
            "WHERE current_setting('pg_gftr.join_enabled', true) IS NOT NULL \\gexec"
        )
    if mode in FUSED_MODES:
        settings += ["SET pg_gftr.fused_enabled=on;", "SET pg_gftr.fused_force=on;"]
        if mode == "fused_parallel":
            settings.append("SET pg_gftr.fused_parallel=on;")
        else:
            # Run the serial control against builds from before parallel fusion.
            settings.append(
                "SELECT 'SET pg_gftr.fused_parallel=off;' "
                "WHERE current_setting('pg_gftr.fused_parallel', true) IS NOT NULL \\gexec"
            )
    else:
        # Older builds have no fused GUC. Native and existing modes still work.
        settings.append(
            "SELECT 'SET pg_gftr.fused_enabled=off;' "
            "WHERE current_setting('pg_gftr.fused_enabled', true) IS NOT NULL \\gexec"
        )
    return settings


def record(name):
    return f"\\echo GFTR_JOIN_RECORD {name}"


def workload(args, database_number):
    uses_extension = any(mode not in NATIVE_MODES for mode in args.modes)
    script = [
        f"SET statement_timeout={args.statement_timeout_ms};",
        "SET search_path=pg_catalog,public;",
    ]
    if uses_extension:
        script += ["LOAD 'pg_gftr';", f"SET pg_gftr.device={args.device};"]
    if any(mode in CUSTOM_JOIN_MODES for mode in args.modes):
        script += [
            f"SET pg_gftr.join_batch_rows={args.join_batch_rows};",
            f"SET pg_gftr.join_max_host_mb={args.join_max_host_mb};",
            f"SET pg_gftr.join_max_device_mb={args.join_max_device_mb};",
        ]
    if "native_gpu_agg" in args.modes:
        script += [
            "SET pg_gftr.backend='cuda';",
            "SET pg_gftr.mode='gftr';",
            "SET pg_gftr.force=on;",
            f"SET pg_gftr.max_input_mb={args.agg_max_input_mb};",
            f"SET pg_gftr.max_device_mb={args.join_max_device_mb};",
        ]
    if args.serial:
        script.append("SET max_parallel_workers_per_gather=0;")
    if args.disable_jit:
        script.append("SET jit=off;")
    keys = ", ".join(f"'{key}'" for key in SETTINGS)
    script += [
        "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;",
        record("environment"),
        "SELECT json_build_object('database', current_database(), 'postgres', version(), "
        f"'settings', (SELECT json_object_agg(name, current_setting(name)) "
        f"FROM pg_settings WHERE name IN ({keys})));",
    ]
    cases = {case: mode_variants(args, case) for case in args.cases if case_modes(args, case)}
    verification = {}
    for case, variants in cases.items():
        verify_variants = {"native": ("native", None), **variants}
        verification[case] = list(verify_variants)
        for label, (mode, workers) in verify_variants.items():
            script += configure(args, mode, workers)
            script += [record(f"result_{case}_{label}"), f"{query_for_mode(case, mode)};"]

    schedule = []
    rng = random.Random(args.seed + database_number)
    for trial in range(args.warmups + args.trials):
        order = [(case, label) for case, variants in cases.items() for label in variants]
        rng.shuffle(order)
        for case, label in order:
            mode, workers = cases[case][label]
            name = f"plan_{trial}_{case}_{label}"
            schedule.append((trial, case, label, name))
            script += configure(args, mode, workers)
            script += [record(name),
                       "EXPLAIN (ANALYZE, BUFFERS, SETTINGS, TIMING OFF, FORMAT JSON)"
                       f" {query_for_mode(case, mode)};"]
    script.append("COMMIT;")
    return "\n".join(script) + "\n", cases, verification, schedule


def walk_nodes(node):
    yield node
    for child in node.get("Plans", []):
        yield from walk_nodes(child)


def check_plan(plan, case, mode):
    nodes = list(walk_nodes(plan["Plan"]))
    joins = [node for node in nodes if node.get("Custom Plan Provider") == "GftrJoin"]
    aggregations = [node for node in nodes
                    if node.get("Custom Plan Provider") == "GftrAggregate"]
    fused = [node for node in nodes
             if node.get("Custom Plan Provider") == "GftrFusedJoinAggregate"]
    if len(fused) != int(mode in FUSED_MODES):
        raise ValueError(f"{case}/{mode} expected a different number of GftrFusedJoinAggregate nodes")
    if fused and fused[0].get("Backend") != "CPU":
        raise ValueError(f"{case}/{mode} expected the CPU backend at GftrFusedJoinAggregate")
    is_parallel = mode == "fused_parallel"
    if fused and bool(fused[0].get("Partial", False)) != is_parallel:
        raise ValueError(f"{case}/{mode} has an unexpected fused partial aggregate setting")
    if is_parallel:
        check_parallel_fusion(nodes, fused[0], case)
    if mode != "auto" and len(joins) != int(mode in ("cpu", "cuda", *FUSED_MODES)):
        raise ValueError(f"{case}/{mode} expected a different number of GftrJoin nodes")
    if len(aggregations) != int(mode == "native_gpu_agg"):
        raise ValueError(f"{case}/{mode} expected a different number of GftrAggregate nodes")
    native_joins = [node for node in nodes if "Join Type" in node and node not in joins]
    if not joins and not native_joins:
        raise ValueError(f"{case}/{mode} has no join node")
    if mode == "native_preagg" and not any(
        child.get("Node Type") == "Aggregate"
        for join in native_joins
        for branch in join.get("Plans", [])
        for child in walk_nodes(branch)
    ):
        raise ValueError(f"{case}/{mode} has no aggregate below a native join")
    for node in joins:
        backend = str(node.get("Backend", "")).lower()
        expected = "cpu" if mode in ("cpu", *FUSED_MODES) else "cuda"
        if expected not in backend:
            raise ValueError(f"{case}/{mode} expected the {expected} backend, got {backend!r}")
    return [{key: value for key, value in node.items() if key != "Plans"}
            for node in [*fused, *joins, *aggregations, *native_joins]]


def check_parallel_fusion(nodes, partial, case):
    for final in nodes:
        if final.get("Node Type") != "Aggregate" or final.get("Partial Mode") != "Finalize":
            continue
        for gather in walk_nodes(final):
            if gather.get("Node Type") not in ("Gather", "Gather Merge"):
                continue
            if any(node is partial for node in walk_nodes(gather)):
                if gather.get("Workers Launched", 0) < 1:
                    raise ValueError(f"{case}/fused_parallel Gather launched no workers")
                return
    raise ValueError(f"{case}/fused_parallel expected Finalize Aggregate above Gather "
                     "and partial GftrFusedJoinAggregate below Gather")


def count_workers(plan, field):
    return sum(node.get(field, 0) for node in walk_nodes(plan["Plan"]))


def summarize(samples):
    return {
        "execution_ms": [sample["Execution Time"] for sample in samples],
        "median_ms": statistics.median(sample["Execution Time"] for sample in samples),
        "min_ms": min(sample["Execution Time"] for sample in samples),
        "max_ms": max(sample["Execution Time"] for sample in samples),
        "workers_planned": [count_workers(sample, "Workers Planned") for sample in samples],
        "workers_launched": [count_workers(sample, "Workers Launched") for sample in samples],
    }


def run_database(args, dbname, database_number, output):
    output.mkdir()
    script, cases, verification, schedule = workload(args, database_number)
    sql_path = output / "workload.sql"
    sql_path.write_text(script)
    stdout_path = output / "stdout.txt"
    stderr_path = output / "stderr.txt"
    cmd = [args.psql, "-X", "-qAt", "-v", "ON_ERROR_STOP=1", "--dbname", dbname,
           "--file", str(sql_path)]
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)
    chunks = RECORD.split(stdout_path.read_text())
    records = {name: value.strip("\n") for name, value in zip(chunks[1::2], chunks[2::2])}
    hashes = {
        case: {mode: hashlib.sha256(records[f"result_{case}_{mode}"].encode()).hexdigest()
               for mode in modes}
        for case, modes in verification.items()
    }
    samples = {case: {label: [] for label in variants} for case, variants in cases.items()}
    plans = []
    for trial, case, label, name in schedule:
        mode, workers = cases[case][label]
        plan, = json.loads(records[name])
        operators = check_plan(plan, case, mode)
        warmup = trial < args.warmups
        if not warmup:
            samples[case][label].append(plan)
        plans.append({"trial": trial + 1, "warmup": warmup, "case": case, "mode": label,
                      "worker_limit": 0 if args.serial else workers,
                      "operators": operators, "explain": plan})
    for case, values in hashes.items():
        if len(set(values.values())) != 1:
            raise ValueError(f"{case} has different result checksums: {values}")
    report = {
        "environment": json.loads(records["environment"]),
        "configuration": {key: value for key, value in vars(args).items()
                          if key not in ("psql", "dbname", "output")},
        "queries": {case: QUERIES[case].strip() for case in verification},
        "checksums_sha256": hashes,
        "measurements": {case: {mode: summarize(values) for mode, values in modes.items()}
                         for case, modes in samples.items()},
        "plans": plans,
        "timing": ("EXPLAIN ANALYZE execution time in a single session and snapshot. "
                   "Includes SQL execution and accelerator transfers; excludes normal "
                   "result delivery. Result digests are collected outside measured trials."),
    }
    if "native_preagg" in args.modes and "supplier_grouped" in verification:
        report["query_overrides"] = {"native_preagg": {
            "supplier_grouped": PREAGG_SUPPLIER_QUERY.strip(),
        }}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    args = arguments()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = args.output / stamp
    root.mkdir(parents=True)
    for number, dbname in enumerate(args.dbname):
        output = root / f"database-{number + 1}"
        print(f"Running database {number + 1}; artifacts in {output}", flush=True)
        try:
            report = run_database(args, dbname, number, output)
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"psql exited with status {error.returncode}; see "
                             f"{output / 'stderr.txt'}") from None
        except (OSError, ValueError, KeyError) as error:
            raise SystemExit(f"Benchmark failed: {error}; see {output}") from error
        for case, modes in report["measurements"].items():
            summary = ", ".join(f"{mode} {result['median_ms']:.1f} ms"
                                for mode, result in modes.items())
            print(f"{report['environment']['database']} {case}: {summary}")
    print(f"Checksums agree. Reports in {root}")


if __name__ == "__main__":
    main()
