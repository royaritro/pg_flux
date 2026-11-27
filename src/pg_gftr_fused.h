#ifndef PG_GFTR_FUSED_H
#define PG_GFTR_FUSED_H

#include "nodes/execnodes.h"
#include "nodes/extensible.h"
#include "nodes/pathnodes.h"

typedef struct GftrFusedGroup
{
	int32 key;
	bool key_null;
	int64 count;
	int64 sum;
	bool sum_valid;
} GftrFusedGroup;

typedef struct GftrFusedResult
{
	GftrFusedGroup *groups;
	int ngroups;
	uint64 probe_rows;
	uint64 match_rows;
	double build_ms;
	double probe_ms;
} GftrFusedResult;

void pg_gftr_fused_prepare_join(CustomScanState *join);

/* Input attribute numbers are indexes into the join's scan slot. A zero
 * sum attribute requests COUNT(*) only. Results belong to context. */
void pg_gftr_fused_reduce(CustomScanState *join, AttrNumber group_attno,
						  AttrNumber sum_attno, MemoryContext context,
						  GftrFusedResult *result);

/* Build a private, serial GftrJoin child from a native join's proof. */
Path *pg_gftr_fused_join_path(PlannerInfo *root, Path *native_path);

bool pg_gftr_add_parallel_fused_path(PlannerInfo *root, RelOptInfo *output_rel,
									GroupPathExtraData *extra, Path *serial_join,
									List *scan_tlist, const CustomPathMethods *methods,
									bool force);

#endif
