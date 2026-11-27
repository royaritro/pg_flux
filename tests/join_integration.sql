\set ON_ERROR_STOP on

LOAD 'pg_gftr';
SET pg_gftr.enabled = off;
SET pg_gftr.join_backend = :'backend';
SET pg_gftr.join_force = on;
SET pg_gftr.join_batch_rows = 17;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

CREATE TEMP TABLE join_build(k int PRIMARY KEY, payload int, label text);
CREATE TEMP TABLE join_probe(k int, payload int, other int);
CREATE TEMP TABLE join_nonunique(k int, payload int);
CREATE TEMP TABLE join_empty(k int PRIMARY KEY, payload int);

INSERT INTO join_build VALUES
    (-2147483648, 7, 'first'), (-9, 10, 'negative'),
    (0, NULL, 'zero'), (2, 20, 'two'), (2147483647, -3, 'last');
INSERT INTO join_probe VALUES
    (0, NULL, 8), (2, 15, 0), (2, 15, 0), (3, 5, 0),
    (NULL, 7, 0), (-2147483648, 2, 0), (2147483647, NULL, 7);
INSERT INTO join_probe
SELECT (n % 7) - 3, n, CASE WHEN n % 3 = 0 THEN NULL ELSE n END
FROM generate_series(1, 227) n;
INSERT INTO join_nonunique VALUES (0, 1), (0, 2), (2, 4), (NULL, 0);
ANALYZE join_build;
ANALYZE join_probe;
ANALYZE join_nonunique;
ANALYZE join_empty;

CREATE FUNCTION pg_temp.assert_join(q text, custom_expected boolean)
RETURNS void LANGUAGE plpgsql AS $$
DECLARE
    plan json;
BEGIN
    PERFORM set_config('pg_gftr.join_enabled', 'off', true);
    EXECUTE 'CREATE TEMP TABLE join_expected AS ' || q;

    PERFORM set_config('pg_gftr.join_enabled', 'on', true);
    EXECUTE 'EXPLAIN (FORMAT JSON) ' || q INTO plan;
    IF (plan::text LIKE '%GftrJoin%') <> custom_expected THEN
        RAISE EXCEPTION 'unexpected join plan for %: %', q, plan;
    END IF;

    EXECUTE 'CREATE TEMP TABLE join_actual AS ' || q;
    IF EXISTS (
        (TABLE join_expected EXCEPT ALL TABLE join_actual)
        UNION ALL
        (TABLE join_actual EXCEPT ALL TABLE join_expected)
    ) THEN
        RAISE EXCEPTION 'join result mismatch for %', q;
    END IF;
    DROP TABLE join_expected, join_actual;
END $$;

SELECT pg_temp.assert_join(
    'SELECT b.payload AS dimension, p.payload AS fact, p.k, p.other FROM join_probe p JOIN join_build b ON p.k = b.k', true);
SELECT pg_temp.assert_join(
    'SELECT p.other, b.payload AS dimension, b.k, p.payload AS fact FROM join_build b JOIN join_probe p ON b.k = p.k WHERE p.payload > 27', true);
SELECT pg_temp.assert_join(
    'SELECT count(*), sum(b.payload) AS dimension, sum(p.other) AS fact FROM join_probe p JOIN join_build b USING (k)', true);
SELECT pg_temp.assert_join(
    'SELECT b.payload AS dimension, p.payload AS fact FROM join_probe p JOIN join_empty b USING (k)', true);
SELECT pg_temp.assert_join(
    'SELECT p.k, b.payload AS dimension, p.payload AS fact FROM join_probe p LEFT JOIN join_build b ON p.k = b.k', false);
SELECT pg_temp.assert_join(
    'SELECT p.payload AS fact, b.payload AS dimension FROM join_probe p JOIN join_nonunique b USING (k)', false);
SELECT pg_temp.assert_join(
    'SELECT p.payload AS fact, b.payload AS dimension FROM join_probe p JOIN join_build b ON p.k = b.k AND p.payload > b.payload', false);
SELECT pg_temp.assert_join(
    'SELECT p.payload, b.label FROM join_probe p JOIN join_build b USING (k)', false);
SELECT pg_temp.assert_join(
    'SELECT p.payload AS fact, b.payload AS dimension FROM join_probe p JOIN join_build b ON p.k::bigint = b.k::bigint', false);

SET pg_gftr.join_enabled = on;
PREPARE probe_query(int) AS
    SELECT count(*), sum(b.payload), sum(p.other)
    FROM join_probe p JOIN join_build b USING (k) WHERE p.payload > $1;
EXECUTE probe_query(17);
EXECUTE probe_query(137);
DEALLOCATE probe_query;

-- A parent Material node can provide scrollability for the forward-only join.
BEGIN;
DECLARE join_cursor SCROLL CURSOR FOR
    SELECT p.k, p.payload, b.payload FROM join_probe p JOIN join_build b USING (k);
FETCH FORWARD 10 FROM join_cursor;
FETCH BACKWARD 5 FROM join_cursor;
CLOSE join_cursor;
COMMIT;

SELECT pg_temp.assert_join(
    'SELECT x, (SELECT count(*) FROM join_probe p JOIN join_build b ON p.k = b.k WHERE p.payload < x) AS hits FROM generate_series(10,50,10) x', true);

DO $$
BEGIN
    PERFORM set_config('pg_gftr.join_max_host_mb', '1', true);
    PERFORM set_config('pg_gftr.join_batch_rows', '1048576', true);
    BEGIN
        EXECUTE 'SELECT p.payload FROM join_probe p JOIN join_build b USING (k)';
        RAISE EXCEPTION 'join host budget was not enforced';
    EXCEPTION WHEN program_limit_exceeded THEN
        NULL;
    END;
    PERFORM set_config('pg_gftr.join_batch_rows', '17', true);
    PERFORM pg_temp.assert_join(
        'SELECT p.payload FROM join_probe p JOIN join_build b USING (k)', true);
END $$;
