/* Parallel fused aggregation using worker-local build and reduction state. */
#include "postgres.h"

#include "nodes/makefuncs.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/planner.h"
#include "optimizer/tlist.h"

#include "pg_gftr_fused.h"

static RelOptInfo *
partial_grouping_rel(PlannerInfo *root, RelOptInfo *output_rel)
{
	ListCell *lc;

	/* PostgreSQL builds the partial aggregation target before this hook. */
	foreach(lc, root->upper_rels[UPPERREL_PARTIAL_GROUP_AGG])
	{
		RelOptInfo *rel = lfirst_node(RelOptInfo, lc);

		if (rel->consider_parallel && bms_equal(rel->relids, output_rel->relids))
			return rel;
	}
	return NULL;
}

static CustomPath *
make_partial_join(PlannerInfo *root, Path *serial_join)
{
	CustomPath *join = castNode(CustomPath, serial_join);
	Path *serial_probe = linitial(join->custom_paths);
	Path *serial_build = lsecond(join->custom_paths);
	Path *build = NULL;
	Path *probe = NULL;
	CustomPath *copy;
	ListCell *lc;

	foreach(lc, serial_probe->parent->partial_pathlist)
	{
		Path *candidate = lfirst(lc);

		/* Use core's shared SeqScan and preserve the input Vars
		 * certified by the ordinary join path. */
		if (candidate->pathtype == T_SeqScan && candidate->parallel_aware &&
			candidate->parallel_safe && candidate->parallel_workers > 0 &&
			!candidate->param_info &&
			equal(candidate->pathtarget->exprs, serial_probe->pathtarget->exprs) &&
			(!probe || candidate->total_cost < probe->total_cost))
			probe = candidate;
	}
	if (!probe)
		return NULL;

	foreach(lc, serial_build->parent->pathlist)
	{
		Path *candidate = lfirst(lc);

		if (candidate->parallel_safe && !candidate->parallel_aware &&
			!candidate->param_info &&
			equal(candidate->pathtarget->exprs, serial_build->pathtarget->exprs) &&
			(!build || compare_path_costs(candidate, build, TOTAL_COST) < 0))
			build = candidate;
	}
	/* A cheap Gather may have displaced the serial scan in the path list.
	 * Each participant still needs an independent complete build scan. */
	if (!build)
	{
		Path *candidate = create_seqscan_path(root, serial_build->parent, NULL, 0);

		if (!candidate->parallel_safe ||
			!equal(candidate->pathtarget->exprs, serial_build->pathtarget->exprs))
			return NULL;
		build = candidate;
	}

	copy = makeNode(CustomPath);
	*copy = *join;
	copy->path.rows = clamp_row_est(serial_join->rows * probe->rows /
									Max(1.0, serial_probe->rows));
	copy->path.parallel_aware = false;
	copy->path.parallel_safe = true;
	copy->path.parallel_workers = probe->parallel_workers;
	copy->path.disabled_nodes = probe->disabled_nodes + build->disabled_nodes;
	copy->path.startup_cost = build->total_cost + probe->startup_cost;
	copy->path.total_cost = build->total_cost + probe->total_cost +
		cpu_operator_cost * (build->rows + probe->rows);
	copy->custom_paths = list_make2(probe, build);
	return copy;
}

bool
pg_gftr_add_parallel_fused_path(PlannerInfo *root, RelOptInfo *output_rel,
								GroupPathExtraData *extra, Path *serial_join,
								List *scan_tlist, const CustomPathMethods *methods,
								bool force)
{
	RelOptInfo *partial_rel;
	CustomPath *join;
	CustomPath *partial;
	GatherPath *gather;
	AggPath *final;
	double groups;
	double gathered_groups;
	ListCell *lc;

	if (root->parent_root || !output_rel->consider_parallel ||
		(extra->flags & (GROUPING_CAN_PARTIAL_AGG | GROUPING_CAN_USE_HASH)) !=
		(GROUPING_CAN_PARTIAL_AGG | GROUPING_CAN_USE_HASH))
		return false;
	partial_rel = partial_grouping_rel(root, output_rel);
	if (!partial_rel || partial_rel->reltarget->exprs == NIL)
		return false;
	join = make_partial_join(root, serial_join);
	if (!join)
		return false;

	/* The worker returns int8 transition states for these two aggregates.
	 * Core matches these expressions when planning the final combine. */
	scan_tlist = copyObject(scan_tlist);
	for_each_from(lc, scan_tlist, 1)
	{
		TargetEntry *entry = lfirst_node(TargetEntry, lc);

		mark_partial_aggref(castNode(Aggref, entry->expr), AGGSPLIT_INITIAL_SERIAL);
	}
	foreach(lc, partial_rel->reltarget->exprs)
		if (!tlist_member(lfirst(lc), scan_tlist))
			return false;

	groups = ((Path *) linitial(output_rel->pathlist))->rows;
	partial = makeNode(CustomPath);
	partial->path.pathtype = T_CustomScan;
	partial->path.parent = partial_rel;
	partial->path.pathtarget = partial_rel->reltarget;
	partial->path.parallel_safe = true;
	partial->path.parallel_workers = join->path.parallel_workers;
	partial->path.disabled_nodes = join->path.disabled_nodes;
	partial->path.rows = Min(groups, join->path.rows);
	partial->path.startup_cost = join->path.total_cost;
	partial->path.total_cost = partial->path.startup_cost +
		cpu_tuple_cost * partial->path.rows;
	partial->custom_paths = list_make1(join);
	partial->custom_private = scan_tlist;
	partial->methods = methods;

	gathered_groups = partial->path.rows * (partial->path.parallel_workers + 1);
	gather = create_gather_path(root, partial_rel, &partial->path,
							  partial->path.pathtarget, NULL, &gathered_groups);
	final = create_agg_path(root, output_rel, &gather->path, output_rel->reltarget,
							AGG_HASHED, AGGSPLIT_FINAL_DESERIAL,
							root->processed_groupClause, NIL, &extra->agg_final_costs, groups);
	if (force)
	{
		final->path.disabled_nodes = 0;
		final->path.startup_cost = 0;
		final->path.total_cost = 0.0001;
	}
	add_path(output_rel, &final->path);
	return true;
}
