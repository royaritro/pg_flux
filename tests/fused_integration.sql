\set ON_ERROR_STOP on
LOAD 'pg_gftr';
SET pg_gftr.enabled = off;
SET pg_gftr.join_backend = 'cpu';
SET pg_gftr.join_enabled = off;
SET pg_gftr.join_force = off;
SET pg_gftr.fused_enabled = on;
SET pg_gftr.fused_force = on;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

CREATE TEMP TABLE fused_dim(id int PRIMARY KEY, grp int);
CREATE TEMP TABLE fused_fact(id int, value int, grp int);
CREATE TEMP TABLE fused_nonunique(id int, grp int);
CREATE TEMP TABLE fused_empty(id int PRIMARY KEY, grp int);
INSERT INTO fused_dim VALUES
    (-2147483648, NULL), (1, -2147483648), (2, -2147483648),
    (3, 2147483647), (4, 2147483647), (5, 5), (6, 0);
INSERT INTO fused_fact VALUES
    (-2147483648, 2, 0), (1, -2147483648, 0), (1, NULL, 0),
    (2, 2147483647, 0), (3, NULL, NULL), (4, NULL, NULL),
    (5, 10, 1), (5, 10, 1), (100, 100, 1), (NULL, 100, 1);
INSERT INTO fused_fact
SELECT 1 + i % 5, CASE WHEN i % 7 = 0 THEN NULL ELSE i END, i % 9
FROM generate_series(1, 501) i;
INSERT INTO fused_nonunique VALUES (1, 2), (1, 3);
ANALYZE fused_dim;
ANALYZE fused_fact;
ANALYZE fused_nonunique;
ANALYZE fused_empty;

CREATE FUNCTION pg_temp.assert_fused(q text, expected boolean, native_only boolean DEFAULT false)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE
    plan json;
BEGIN
    PERFORM set_config('pg_gftr.join_enabled', 'off', true);
    PERFORM set_config('pg_gftr.fused_enabled', 'off', true);
    EXECUTE 'CREATE TEMP TABLE fused_expected AS ' || q;

    PERFORM set_config('pg_gftr.fused_enabled', 'on', true);
    EXECUTE 'EXPLAIN (FORMAT JSON) ' || q INTO plan;
    IF (plan::text LIKE '%GftrFusedJoinAggregate%') <> expected OR
        (native_only AND plan::text LIKE '%"Custom Plan Provider"%')
    THEN
        RAISE EXCEPTION 'unexpected fused join plan for %: %', q, plan;
    END IF;

    EXECUTE 'CREATE TEMP TABLE fused_actual AS ' || q;
    IF EXISTS (
        (TABLE fused_expected EXCEPT ALL TABLE fused_actual)
        UNION ALL
        (TABLE fused_actual EXCEPT ALL TABLE fused_expected)
    ) THEN
        RAISE EXCEPTION 'fused join result mismatch for %', q;
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_attribute e JOIN pg_attribute a USING (attnum)
        WHERE e.attrelid = 'fused_expected'::regclass
          AND a.attrelid = 'fused_actual'::regclass
          AND e.attnum > 0 AND e.atttypid <> a.atttypid
    ) THEN
        RAISE EXCEPTION 'fused join result type mismatch for %', q;
    END IF;
    DROP TABLE fused_expected, fused_actual;
END $$;

SELECT pg_temp.assert_fused(
    'SELECT d.grp, count(*), sum(f.value) FROM fused_fact f JOIN fused_dim d USING (id) GROUP BY d.grp ORDER BY d.grp NULLS FIRST', true);
SELECT pg_temp.assert_fused(
    'SELECT count(*), d.grp, sum(f.value) FROM fused_dim d JOIN fused_fact f USING (id) WHERE f.value IS NULL GROUP BY d.grp', true);
SELECT pg_temp.assert_fused(
    'SELECT d.grp, count(*) FROM fused_fact f JOIN fused_dim d USING (id) WHERE f.value > 40 GROUP BY d.grp', true);
SELECT pg_temp.assert_fused(
    'SELECT d.grp, count(*), sum(f.value) FROM fused_fact f JOIN fused_empty d USING (id) GROUP BY d.grp', true);
SELECT pg_temp.assert_fused(
    'SELECT d.grp, count(*) FROM fused_fact f LEFT JOIN fused_dim d USING (id) GROUP BY d.grp', false, true);
SELECT pg_temp.assert_fused(
    'SELECT d.grp, count(*) FROM fused_fact f JOIN fused_nonunique d USING (id) GROUP BY d.grp', false, true);
SELECT pg_temp.assert_fused(
    'SELECT d.grp, count(*), sum(d.id) FROM fused_fact f JOIN fused_dim d USING (id) GROUP BY d.grp', false, true);
SELECT pg_temp.assert_fused(
    'SELECT d.grp, count(*), sum(f.value) FILTER (WHERE f.value > 0) FROM fused_fact f JOIN fused_dim d USING (id) GROUP BY d.grp', false, true);
SELECT pg_temp.assert_fused(
    'SELECT f.grp, count(*) FROM fused_fact f JOIN fused_dim d USING (id) GROUP BY f.grp', false, true);

SELECT pg_temp.assert_fused(
    'SELECT d.grp, count(*) AS c1, count(*) AS c2, sum(f.value) FROM fused_fact f JOIN fused_dim d USING (id) GROUP BY d.grp ORDER BY c1 DESC, d.grp LIMIT 2', true);
SELECT pg_temp.assert_fused(
    'SELECT x, (SELECT sum(hits) FROM (SELECT d.grp, count(*) AS hits FROM fused_fact f JOIN fused_dim d USING (id) WHERE f.value < x GROUP BY d.grp ORDER BY d.grp LIMIT 2) g) AS groups FROM generate_series(10,50,10) x', true);

SET plan_cache_mode = force_generic_plan;
PREPARE fused_query(int) AS
    SELECT d.grp, count(*), sum(f.value)
    FROM fused_fact f JOIN fused_dim d USING (id)
    WHERE f.value < $1 GROUP BY d.grp ORDER BY d.grp;
EXPLAIN (COSTS OFF) EXECUTE fused_query(42);
SET pg_gftr.join_backend = 'cuda';
EXECUTE fused_query(42);
EXECUTE fused_query(105);
DEALLOCATE fused_query;
