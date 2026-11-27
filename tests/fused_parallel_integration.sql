\set ON_ERROR_STOP on
LOAD 'pg_gftr';

SET min_parallel_table_scan_size = 0;
SET parallel_setup_cost = 0;
SET parallel_tuple_cost = 0;
SET max_parallel_workers_per_gather = 2;
SET jit = off;

-- Parallel workers need permanent relations.
CREATE TABLE fused_parallel_dim(id int PRIMARY KEY, grp int);
CREATE TABLE fused_parallel_fact(id int, value int);
INSERT INTO fused_parallel_dim VALUES (1, 10), (2, 10), (3, NULL), (4, -1), (5, 5);
INSERT INTO fused_parallel_fact
SELECT CASE WHEN n % 19 = 0 THEN NULL ELSE 1 + n % 6 END,
       CASE WHEN n % 2 = 0 THEN NULL ELSE (n % 9) - 4 END
FROM generate_series(1, 40000) n;
ALTER TABLE fused_parallel_fact SET (parallel_workers = 2);
ANALYZE fused_parallel_dim;
ANALYZE fused_parallel_fact;

CREATE FUNCTION pg_temp.assert_parallel_fused(q text, expect_parallel boolean)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE
    plan jsonb;
    launched boolean;
BEGIN
    PERFORM set_config('pg_gftr.fused_enabled', 'off', true);
    EXECUTE 'CREATE TEMP TABLE fused_parallel_expected AS ' || q;
    PERFORM set_config('pg_gftr.fused_enabled', 'on', true);

    EXECUTE 'EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, FORMAT JSON) ' || q INTO plan;
    launched := jsonb_path_exists(plan,
        '$[0].Plan.** ? (@."Node Type" == "Gather" && @."Workers Launched" > 0)');
    IF NOT jsonb_path_exists(plan,
        '$[0].Plan.** ? (@."Custom Plan Provider" == "GftrFusedJoinAggregate")')
        OR launched <> expect_parallel
        OR (expect_parallel AND NOT jsonb_path_exists(plan,
            '$[0].Plan.** ? (@."Partial Mode" == "Finalize")'))
        OR jsonb_path_exists(plan,
            '$[0].Plan.** ? (@."Custom Plan Provider" == "GftrFusedJoinAggregate" && @."Partial" == true)')
            <> expect_parallel
    THEN
        RAISE EXCEPTION 'unexpected parallel fused plan: %', plan;
    END IF;

    EXECUTE 'CREATE TEMP TABLE fused_parallel_actual AS ' || q;
    IF EXISTS (
        (TABLE fused_parallel_expected EXCEPT ALL TABLE fused_parallel_actual)
        UNION ALL
        (TABLE fused_parallel_actual EXCEPT ALL TABLE fused_parallel_expected)
    ) THEN
        RAISE EXCEPTION 'parallel fused result mismatch for %', q;
    END IF;
    DROP TABLE fused_parallel_expected, fused_parallel_actual;
END $$;

SET pg_gftr.join_backend = 'cuda';
SET pg_gftr.fused_force = on;
SET pg_gftr.fused_parallel = on;
SELECT pg_temp.assert_parallel_fused(
    'SELECT d.grp, count(*), sum(f.value) FROM fused_parallel_fact f JOIN fused_parallel_dim d USING (id) GROUP BY d.grp', true);
SELECT pg_temp.assert_parallel_fused(
    'SELECT count(*), d.grp, sum(f.value) FROM fused_parallel_fact f JOIN fused_parallel_dim d USING (id) WHERE f.value IS NULL GROUP BY d.grp', true);
SELECT pg_temp.assert_parallel_fused(
    'SELECT d.grp, count(*) FROM fused_parallel_fact f JOIN fused_parallel_dim d USING (id) GROUP BY d.grp', true);

SET parallel_leader_participation = off;
SELECT pg_temp.assert_parallel_fused(
    'SELECT d.grp, count(*), sum(f.value) FROM fused_parallel_fact f JOIN fused_parallel_dim d USING (id) GROUP BY d.grp', true);
RESET parallel_leader_participation;

SET max_parallel_workers_per_gather = 0;
SELECT pg_temp.assert_parallel_fused(
    'SELECT d.grp, count(*), sum(f.value) FROM fused_parallel_fact f JOIN fused_parallel_dim d USING (id) GROUP BY d.grp', false);
