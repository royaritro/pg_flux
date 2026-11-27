#include "gftr_join.h"
#include "gftr_join_lookup.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct GftrJoinBuild
{
	GftrJoinLayout layout;
	void *table;
};

static double
join_now_ms(void)
{
	struct timespec time;

	clock_gettime(CLOCK_MONOTONIC, &time);
	return time.tv_sec * 1000.0 + time.tv_nsec / 1000000.0;
}

void
gftr_join_cpu_free(GftrJoinBuild *build)
{
	if (build)
	{
		free(build->table);
		free(build);
	}
}

int
gftr_join_cpu_build_groups(const int32_t *keys, const int32_t *groups,
						   int32_t count, size_t budget, GftrJoinBuild **out,
						   GftrJoinTiming *timing, char error[GFTR_ERROR_SIZE])
{
	const char *message;
	GftrJoinLayout layout;
	GftrJoinBuild *build = NULL;
	double start = join_now_ms();
	int32_t i;

	*out = NULL;
	memset(timing, 0, sizeof(*timing));
	if (count < 0 || (count > 0 && !keys))
	{
		message = "invalid CPU join build input";
		goto fail;
	}
	if (gftr_join_layout(keys, count, &layout) || layout.bytes > budget)
	{
		message = "CPU join lookup exceeds the host memory budget";
		goto fail;
	}

	message = "could not allocate CPU join lookup";
	build = calloc(1, sizeof(*build));
	if (!build)
		goto fail;
	build->layout = layout;
	if (layout.bytes > 0)
	{
		build->table = malloc(layout.bytes);
		if (!build->table)
			goto fail;
		memset(build->table, 0xff, layout.bytes);
	}

	message = "duplicate key on the unique join build side";
	if (layout.dense)
	{
		int32_t *table = build->table;

		for (i = 0; i < count; i++)
		{
			size_t slot = (size_t) ((int64_t) keys[i] - layout.minimum);

			if (table[slot] != -1)
				goto fail;
			table[slot] = groups ? groups[i] : i;
		}
	}
	else
	{
		GftrJoinEntry *table = build->table;
		size_t mask = layout.slots - 1;

		for (i = 0; i < count; i++)
		{
			size_t slot = gftr_join_hash(keys[i], mask);

			while (table[slot] != GFTR_JOIN_EMPTY)
			{
				if (gftr_join_is_key(table[slot], keys[i]))
					goto fail;
				slot = (slot + 1) & mask;
			}
			table[slot] = gftr_join_entry(keys[i], groups ? groups[i] : i);
		}
	}

	timing->build_ms = join_now_ms() - start;
	timing->build_bytes = layout.bytes;
	timing->algorithm = count == 0 ? "empty" : layout.dense ? "dense" : "hash";
	*out = build;
	return 0;

fail:
	gftr_join_cpu_free(build);
	snprintf(error, GFTR_ERROR_SIZE, "%s", message);
	return -1;
}

int
gftr_join_cpu_build(const int32_t *keys, int32_t count, size_t budget,
					GftrJoinBuild **out, GftrJoinTiming *timing,
					char error[GFTR_ERROR_SIZE])
{
	return gftr_join_cpu_build_groups(keys, NULL, count, budget, out, timing, error);
}

int32_t
gftr_join_cpu_lookup(const GftrJoinBuild *build, int32_t key)
{
	if (build->layout.dense)
		return gftr_join_dense_lookup(build->table, build->layout, key);
	return gftr_join_hash_lookup(build->table, build->layout.slots, key);
}

int
gftr_join_cpu_probe(GftrJoinBuild *build, const int32_t *keys,
					const uint8_t *valid, int32_t count, int32_t *matches,
					GftrJoinTiming *timing, char error[GFTR_ERROR_SIZE])
{
	double start = join_now_ms();
	int32_t i;

	if (!build || count < 0 || (count > 0 && (!keys || !matches)))
	{
		snprintf(error, GFTR_ERROR_SIZE, "invalid CPU join probe input");
		return -1;
	}

	if (build->layout.dense)
	{
		for (i = 0; i < count; i++)
			matches[i] = !valid || valid[i] ?
				gftr_join_dense_lookup(build->table, build->layout, keys[i]) : -1;
	}
	else
	{
		for (i = 0; i < count; i++)
			matches[i] = !valid || valid[i] ?
				gftr_join_hash_lookup(build->table, build->layout.slots, keys[i]) : -1;
	}

	timing->probe_ms += join_now_ms() - start;
	return 0;
}
