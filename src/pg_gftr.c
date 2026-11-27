/* PostgreSQL 18 custom upper path for single-int4-key aggregation. */
#include "postgres.h"

#include "access/htup_details.h"
#include "catalog/pg_aggregate.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "commands/explain.h"
#include "commands/explain_format.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/planner.h"
#include "optimizer/tlist.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "portability/instr_time.h"

#include "gftr_backend.h"
#include "gftr_join.h"
#include "pg_gftr_fused.h"

#if PG_VERSION_NUM < 180000 || PG_VERSION_NUM >= 190000
#error "pg_gftr currently supports PostgreSQL 18 only"
#endif

PG_MODULE_MAGIC;
PG_FUNCTION_INFO_V1(pg_gftr_version);
void _PG_init(void);
void _PG_fini(void);

/**
 * ENUM representing supported aggregate operations.
 */
enum GftrOp
{
	GFTR_COUNT_STAR = 1, // COUNT(*)
	GFTR_COUNT,          // COUNT(column)
	GFTR_SUM,            // SUM(column)
	GFTR_MIN,            // MIN(column)
	GFTR_MAX             // MAX(column)
};

/**
 * GftrState holds the runtime state of one pg_gftr custom scan node. This is equivalent to in OS terms, GftrState is the Process Control Block (PCB).
 */
typedef struct GftrState
{
	/**
	 * SQL text
  		→ parser    → Query tree  (what you asked for)
  		→ planner   → Plan tree   (how to do it)
  		→ executor  → PlanState tree (one node per Plan node, holding run-time state)
	 * CustomScanState stores ScanState which stores PlanState.
	 * ScanState is PlanState plus what this node is reading (a table and a cursor into it) and a slot for the current row.
	 * CustomScanState extends ScanState with additional fields for custom scan nodes.
	 * CustomScanState is ScanState plus "my code lives in an extension" (methods), It means that PostgreSQL doesn't contain your node's code. Your .so does, and methods is how PostgreSQL finds that code., "here are my children" (custom_ps), custom node can have any number of children, and what I support (flags, pscan_len, slotOps).
	 */
	CustomScanState css;
	/**
	 * Pointer to the child PlanState of this custom scan node. We are doing aggregation and aggregation just has 1 child.
	 */
	PlanState *child;
	/**
	 * A MemoryContext is a memory pool. Everything you palloc (palloc is Postgres' own memory management system) goes into the current context, and when that context is reset or deleted, all of it is freed at once. You don't have to free each allocation individually. How it works inside: the pool calls malloc for big blocks, say 8 KB at a time. Each palloc then hands out the next slice of the current block: if the current block is full, it calls malloc for a new block and continues handing out slices from the new block.
	 */
	MemoryContext context;
	/**
	 * MemoryContextCallback is a request of the form "when this memory context is reset or deleted, call this function first". It is one node in a linked list of callbacks associated with the memory context.
	 */
	MemoryContextCallback cleanup;
	/**
	 * key_attno: key_attno is the column number of the GROUP BY key, ncols is the number of distinct value columns the aggregates need, and attnos is the array of column numbers for those value columns.
	 * CREATE TABLE sales (region int, store int, qty int, price int, discount int);
	 * SELECT region, sum(price), max(qty), min(price)
	   FROM sales GROUP BY region;
	 * child row:   region | store | qty | price | discount
		   attno:      1   |   2   |  3  |   4   |    5
	 * key_attno = 1 (region)
	 * ncols = 3 (sum(price), max(qty), min(price))
	 * attnos = {4, 3, 4} (price, qty, price)
	 */
	int key_attno; // Column number (1-based) of the GROUP BY key in the child's output row
	int ncols; // How many distinct value columns the aggregates need
	int *attnos; // Array of ncols column numbers, telling you where each of those value columns sits in the child's row
	/**
	 * ops is a list of lists. Each inner list is a pair of ints: (Gftr_op, attno)
	 * SELECT region, count(*), sum(price), max(qty), min(price) FROM sales GROUP BY region;
	 * ops = [ 
	 * [COUNT_STAR, -1],    count(*)
	 * [SUM,         0],    sum(price) → columns[0]
	 * [MAX,         1],    max(qty)   → columns[1]
	 * [MIN,         0] ]   min(price) → columns[0]
	 */
	List *ops;
	GftrInput input; // Go to it's .h
	int capacity;
	int max_rows;
	GftrResult result;
	int next_group;
	bool loaded;
	int backend;
	int mode;
	int device;
	size_t device_bytes;
	double pack_ms;
} GftrState;

typedef struct GftrFusedAggregateState
{
	CustomScanState css;
	CustomScanState *join;
	MemoryContext result_context;
	List *ops;
	AttrNumber group_attno;
	AttrNumber sum_attno;
	bool loaded;
	int next_group;
	GftrFusedResult result;
} GftrFusedAggregateState;

static bool fused_enabled = false;
static bool fused_force = false;
static bool fused_parallel = false;
static bool gftr_enabled = false;
static bool gftr_force = false;
static int gftr_backend = 0;
static int gftr_mode = 0;
static int gftr_device = 0;
static int gftr_min_rows = 100000;
static int gftr_max_input_mb = 256;
static int gftr_max_device_mb = 4096;
static double gftr_setup_cost = 1000.0;
static double gftr_tuple_cost = 0.02;
static create_upper_paths_hook_type previous_upper_hook = NULL;

static const struct config_enum_entry backend_options[] = {
	{"cuda", 0, false}, {"cpu", 1, false}, {NULL, 0, false}
};
static const struct config_enum_entry mode_options[] = {
	{"gftr", 0, false}, {"gfur", 1, false}, {NULL, 0, false}
};

static void gftr_upper_paths(PlannerInfo *, UpperRelationKind,
							 RelOptInfo *, RelOptInfo *, void *);
static Plan *gftr_plan_path(PlannerInfo *, RelOptInfo *, CustomPath *,
							List *, List *, List *);
static Node *gftr_create_state(CustomScan *);
static void gftr_begin(CustomScanState *, EState *, int);
static TupleTableSlot *gftr_exec(CustomScanState *);
static void gftr_end(CustomScanState *);
static void gftr_rescan(CustomScanState *);
static void gftr_explain(CustomScanState *, List *, ExplainState *);

static Node *fused_create_state(CustomScan *);
static void fused_begin(CustomScanState *, EState *, int);
static TupleTableSlot *fused_exec(CustomScanState *);
static void fused_end(CustomScanState *);
static void fused_rescan(CustomScanState *);
static void fused_explain(CustomScanState *, List *, ExplainState *);

static const CustomPathMethods path_methods = {
	.CustomName = "GftrAggregate",
	.PlanCustomPath = gftr_plan_path
};
static const CustomScanMethods scan_methods = {
	.CustomName = "GftrAggregate",
	.CreateCustomScanState = gftr_create_state
};
static const CustomExecMethods exec_methods = {
	.CustomName = "GftrAggregate",
	.BeginCustomScan = gftr_begin,
	.ExecCustomScan = gftr_exec,
	.EndCustomScan = gftr_end,
	.ReScanCustomScan = gftr_rescan,
	.ExplainCustomScan = gftr_explain
};

static const CustomPathMethods fused_path_methods = {
	.CustomName = "GftrFusedJoinAggregate",
	.PlanCustomPath = gftr_plan_path
};
static const CustomScanMethods fused_scan_methods = {
	.CustomName = "GftrFusedJoinAggregate",
	.CreateCustomScanState = fused_create_state
};
static const CustomExecMethods fused_exec_methods = {
	.CustomName = "GftrFusedJoinAggregate",
	.BeginCustomScan = fused_begin,
	.ExecCustomScan = fused_exec,
	.EndCustomScan = fused_end,
	.ReScanCustomScan = fused_rescan,
	.ExplainCustomScan = fused_explain
};

Datum
pg_gftr_version(PG_FUNCTION_ARGS)
{
#ifdef GFTR_WITH_CUDA
	PG_RETURN_TEXT_P(cstring_to_text("pg_gftr 0.1.0 (CUDA + CPU reference)"));
#else
	PG_RETURN_TEXT_P(cstring_to_text("pg_gftr 0.1.0 (CPU reference; CUDA not built)"));
#endif
}

void
_PG_init(void)
{
	DefineCustomBoolVariable("pg_gftr.fused_enabled", "Offer CPU fused join and grouped aggregate paths.",
		NULL, &fused_enabled, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_gftr.fused_force", "Prefer supported fused paths for experiments.",
		NULL, &fused_force, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_gftr.fused_parallel", "Offer worker-local fused aggregation.",
		NULL, &fused_parallel, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_gftr.enabled", "Offer GPU aggregate paths.",
		NULL, &gftr_enabled, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_gftr.force", "Prefer supported custom paths for experiments.",
		NULL, &gftr_force, false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("pg_gftr.backend", "Execution backend; CPU is a correctness reference.",
		NULL, &gftr_backend, 0, backend_options, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("pg_gftr.mode", "GFTR transforms payloads; GFUR gathers by tuple ID.",
		NULL, &gftr_mode, 0, mode_options, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_gftr.device", "CUDA device ordinal in this process.",
		NULL, &gftr_device, 0, 0, 1024, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_gftr.min_rows", "Minimum estimated input rows unless force is on.",
		NULL, &gftr_min_rows, 100000, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_gftr.max_input_mb", "Limit for packed host input in MB.",
		NULL, &gftr_max_input_mb, 256, 1, 16384, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_gftr.max_device_mb", "Limit for this operator's device allocations in MB.",
		NULL, &gftr_max_device_mb, 4096, 1, 1048576, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("pg_gftr.setup_cost", "Estimated GPU setup cost in planner units.",
		NULL, &gftr_setup_cost, 1000.0, 0.0, 1e12, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("pg_gftr.tuple_cost", "Estimated packing, transfer and compute cost per row.",
		NULL, &gftr_tuple_cost, 0.02, 0.0, 1e6, PGC_USERSET, 0, NULL, NULL, NULL);
	gftr_join_init();
	MarkGUCPrefixReserved("pg_gftr");
	RegisterCustomScanMethods(&scan_methods);
	RegisterCustomScanMethods(&fused_scan_methods);
	previous_upper_hook = create_upper_paths_hook;
	create_upper_paths_hook = gftr_upper_paths;
}

void
_PG_fini(void)
{
	gftr_join_fini();
	if (create_upper_paths_hook == gftr_upper_paths)
		create_upper_paths_hook = previous_upper_hook;
}

/* Match built-in aggregate OIDs, never names that a search_path can shadow. */
static int
aggregate_op(Aggref *agg, AggSplit split)
{
	TargetEntry *arg;
	if (agg->agglevelsup != 0 || agg->aggsplit != split ||
		agg->aggkind != AGGKIND_NORMAL || agg->aggdistinct || agg->aggorder ||
		agg->aggfilter || agg->aggdirectargs || agg->aggvariadic)
		return 0;
	if (agg->aggfnoid == F_COUNT_ && agg->aggstar && agg->args == NIL)
		return GFTR_COUNT_STAR;
	if (list_length(agg->args) != 1)
		return 0;
	arg = linitial_node(TargetEntry, agg->args);
	if (arg->resjunk || !IsA(arg->expr, Var) ||
		exprType((Node *) arg->expr) != INT4OID ||
		((Var *) arg->expr)->varlevelsup != 0)
		return 0;
	switch (agg->aggfnoid)
	{
		case F_COUNT_ANY: return GFTR_COUNT;
		case F_SUM_INT4: return GFTR_SUM;
		case F_MIN_INT4: return GFTR_MIN;
		case F_MAX_INT4: return GFTR_MAX;
		default: return 0;
	}
}

/*
 * The fused reducer consumes the join's packed inputs directly. Resolve
 * attributes against the bare join, even when GROUP BY planning previously
 * wrapped it in a projection path to assign sort/group references.
 */
static Path *
fused_join_child(Path *path, Var *group, Var *sum)
{
	CustomPath *join;
	Path *probe;
	Path *build;

	while (IsA(path, ProjectionPath))
		path = ((ProjectionPath *) path)->subpath;
	if (!IsA(path, CustomPath) || path->param_info)
		return NULL;
	join = (CustomPath *) path;
	if (strcmp(join->methods->CustomName, "GftrJoin") != 0 ||
		list_length(join->custom_paths) != 2)
		return NULL;
	probe = linitial(join->custom_paths);
	build = lsecond(join->custom_paths);
	if (!bms_is_member(group->varno, build->parent->relids) ||
		!list_member(path->pathtarget->exprs, group) ||
		(sum && (!bms_is_member(sum->varno, probe->parent->relids) ||
				 !list_member(path->pathtarget->exprs, sum))))
		return NULL;
	return path;
}

static void
add_fused_path(PlannerInfo *root, RelOptInfo *input_rel, RelOptInfo *output_rel,
			   GroupPathExtraData *extra, List *scan_tlist)
{
	Var *group = (Var *) linitial_node(TargetEntry, scan_tlist)->expr;
	Var *sum = NULL;
	bool count = false;
	Path *best = NULL;
	CustomPath *path;
	ListCell *lc;

	if (group->varattno <= 0 || !bms_is_empty(group->varnullingrels))
		return;
	for_each_from(lc, scan_tlist, 1)
	{
		Aggref *agg = (Aggref *) lfirst_node(TargetEntry, lc)->expr;
		int op = aggregate_op(agg, AGGSPLIT_SIMPLE);

		if (op == GFTR_COUNT_STAR)
			count = true;
		else if (op == GFTR_SUM && !sum)
			sum = (Var *) linitial_node(TargetEntry, agg->args)->expr;
		else
			return;
	}
	if (!count || (sum && (sum->varattno <= 0 || !bms_is_empty(sum->varnullingrels))))
		return;

	foreach(lc, input_rel->pathlist)
	{
		Path *candidate = fused_join_child(lfirst(lc), group, sum);

		if (candidate && (!best || compare_path_costs(candidate, best, TOTAL_COST) < 0))
			best = candidate;
	}
	/*
	 * Fusion is independent of the plain join GUCs. If there is no
	 * suitable custom candidate, use a private child derived from a
	 * native path without adding that child to the join relation.
	 */
	if (!best)
	{
		foreach(lc, input_rel->pathlist)
		{
			Path *candidate = pg_gftr_fused_join_path(root, lfirst(lc));

			if (candidate)
				candidate = fused_join_child(candidate, group, sum);
			if (candidate && (!best || compare_path_costs(candidate, best, TOTAL_COST) < 0))
				best = candidate;
		}
	}
	if (!best || (!fused_force && best->rows < gftr_min_rows))
		return;
	if (fused_parallel &&
		pg_gftr_add_parallel_fused_path(root, output_rel, extra, best, scan_tlist,
										&fused_path_methods, fused_force) &&
		fused_force)
		return;

	path = makeNode(CustomPath);
	path->path.pathtype = T_CustomScan;
	path->path.parent = output_rel;
	path->path.pathtarget = output_rel->reltarget;
	path->path.rows = ((Path *) linitial(output_rel->pathlist))->rows;
	path->path.disabled_nodes = best->disabled_nodes;
	/* The fused executor consumes all probe rows before producing any group. */
	path->path.startup_cost = best->total_cost + cpu_operator_cost * best->rows;
	path->path.total_cost = path->path.startup_cost + cpu_tuple_cost * path->path.rows;
	if (fused_force)
	{
		path->path.disabled_nodes = 0;
		path->path.startup_cost = 0;
		path->path.total_cost = 0.0001;
	}
	path->custom_paths = list_make1(best);
	path->custom_private = scan_tlist;
	path->methods = &fused_path_methods;
	add_path(output_rel, &path->path);
}

static void
gftr_upper_paths(PlannerInfo *root, UpperRelationKind stage,
				 RelOptInfo *input_rel, RelOptInfo *output_rel, void *extra)
{
	GroupPathExtraData *group_extra = extra;
	SortGroupClause *clause;
	Node *key;
	Path *child;
	CustomPath *path;
	List *scan_tlist = NIL;
	ListCell *lc;
	bool has_aggregate = false;
	int resno = 1;

	if (previous_upper_hook)
		previous_upper_hook(root, stage, input_rel, output_rel, extra);
	if ((!gftr_enabled && !fused_enabled) || stage != UPPERREL_GROUP_AGG ||
		!root->parse->hasAggs || list_length(root->parse->groupClause) != 1 ||
		root->parse->groupingSets || root->parse->hasTargetSRFs ||
		root->parse->hasWindowFuncs || group_extra->havingQual ||
		group_extra->patype != PARTITIONWISE_AGGREGATE_NONE ||
		output_rel->pathlist == NIL)
		return;
	clause = linitial_node(SortGroupClause, root->parse->groupClause);
	key = get_sortgroupclause_expr(clause, root->parse->targetList);
	if (!IsA(key, Var) || exprType(key) != INT4OID ||
		((Var *) key)->varlevelsup != 0 || clause->eqop != Int4EqualOperator)
		return;
	scan_tlist = lappend(scan_tlist,
		makeTargetEntry((Expr *) copyObject(key), resno++, NULL, false));
	foreach(lc, output_rel->reltarget->exprs)
	{
		Node *expr = lfirst(lc);
		if (equal(expr, key))
			continue;
		if (!IsA(expr, Aggref) || !aggregate_op((Aggref *) expr, AGGSPLIT_SIMPLE))
			return;
		if (!tlist_member((Expr *) expr, scan_tlist))
			scan_tlist = lappend(scan_tlist,
				makeTargetEntry((Expr *) copyObject(expr), resno++, NULL, false));
		has_aggregate = true;
	}
	if (!has_aggregate)
		return;
	if (fused_enabled)
		add_fused_path(root, input_rel, output_rel, group_extra, scan_tlist);
	if (!gftr_enabled)
		return;

	child = input_rel->cheapest_total_path;
	if (!child || child->param_info ||
		(!gftr_force && child->rows < gftr_min_rows) ||
		!list_member(child->pathtarget->exprs, key))
		return;
	for_each_from(lc, scan_tlist, 1)
	{
		Aggref *agg = (Aggref *) lfirst_node(TargetEntry, lc)->expr;

		if (agg->args != NIL &&
			!list_member(child->pathtarget->exprs,
						 linitial_node(TargetEntry, agg->args)->expr))
			return;
	}
	path = makeNode(CustomPath);
	path->path.pathtype = T_CustomScan;
	path->path.parent = output_rel;
	path->path.pathtarget = output_rel->reltarget;
	path->path.param_info = NULL;
	path->path.parallel_aware = false;
	path->path.parallel_safe = false;
	path->path.parallel_workers = 0;
	path->path.pathkeys = NIL;
	/* Upper relations do not necessarily set RelOptInfo.rows. */
	path->path.rows = ((Path *) linitial(output_rel->pathlist))->rows;
	path->path.startup_cost = child->total_cost + gftr_setup_cost +
		child->rows * gftr_tuple_cost;
	path->path.total_cost = path->path.startup_cost +
		path->path.rows * cpu_tuple_cost;
	/* Explicit experimental override; no claim of calibrated GPU costs. */
	if (gftr_force)
	{
		path->path.startup_cost = 0;
		path->path.total_cost = 0.001;
	}
	path->custom_paths = list_make1(child);
	path->custom_private = scan_tlist;
	path->methods = &path_methods;
	add_path(output_rel, &path->path);
}

static Plan *
gftr_plan_path(PlannerInfo *root, RelOptInfo *rel, CustomPath *path,
			   List *tlist, List *clauses, List *children)
{
	CustomScan *scan = makeNode(CustomScan);
	Plan *child = (Plan *) linitial(children);
	List *scan_tlist = path->custom_private;
	bool fused = path->methods == &fused_path_methods;
	AggSplit split = fused && path->path.parallel_workers > 0 ?
		AGGSPLIT_INITIAL_SERIAL : AGGSPLIT_SIMPLE;
	List *source_tlist = child->targetlist;
	TargetEntry *key_entry;
	List *attrs = NIL;
	List *ops = NIL;
	ListCell *lc;

	if (fused)
	{
		CustomScan *join = castNode(CustomScan, child);

		source_tlist = join->custom_scan_tlist;
	}
	key_entry = tlist_member(linitial_node(TargetEntry, scan_tlist)->expr,
							 source_tlist);
	if (!key_entry)
		elog(ERROR, "pg_gftr: grouping key missing from child target list");
	for_each_from(lc, scan_tlist, 1)
	{
		Aggref *agg = (Aggref *) lfirst_node(TargetEntry, lc)->expr;
		int op = aggregate_op(agg, split);
		int col = -1;
		if (!op)
			elog(ERROR, "pg_gftr: unsupported aggregate in custom plan");
		if (op != GFTR_COUNT_STAR)
		{
			TargetEntry *arg = linitial_node(TargetEntry, agg->args);
			TargetEntry *entry = tlist_member(arg->expr, source_tlist);
			ListCell *ac;
			if (!entry)
				elog(ERROR, "pg_gftr: aggregate input missing from child target list");
			col = 0;
			foreach(ac, attrs)
			{
				if (lfirst_int(ac) == entry->resno)
					break;
				col++;
			}
			if (col == list_length(attrs))
				attrs = lappend_int(attrs, entry->resno);
		}
		ops = lappend(ops, list_make2_int(op, col));
	}
	scan->scan.scanrelid = 0;
	scan->scan.plan.targetlist = tlist;
	scan->custom_scan_tlist = copyObject(scan_tlist);
	scan->custom_plans = children;
	scan->custom_private = list_make3(makeInteger(key_entry->resno), attrs, ops);
	scan->methods = fused ? &fused_scan_methods : &scan_methods;
	return &scan->scan.plan;
}

static Node *
gftr_create_state(CustomScan *scan)
{
	GftrState *state = palloc0(sizeof(GftrState));
	NodeSetTag(&state->css, T_CustomScanState);
	state->css.methods = &exec_methods;
	return (Node *) state;
}

static void
gftr_cleanup(void *arg)
{
	GftrState *state = arg;
	gftr_result_free(&state->result);
}

static void
gftr_begin(CustomScanState *node, EState *estate, int eflags)
{
	GftrState *state = (GftrState *) node;
	CustomScan *plan = (CustomScan *) node->ss.ps.plan;
	List *attrs = lsecond(plan->custom_private);
	ListCell *lc;
	uint64 row_bytes;
	uint64 max_rows;
	int c = 0;

	state->context = estate->es_query_cxt;
	state->key_attno = intVal(linitial(plan->custom_private));
	state->ncols = list_length(attrs);
	state->attnos = palloc(sizeof(int) * Max(1, state->ncols));
	foreach(lc, attrs)
		state->attnos[c++] = lfirst_int(lc);
	state->ops = lthird(plan->custom_private);
	state->input.ncols = state->ncols;
	state->input.columns = palloc0(sizeof(GftrValue *) * Max(1, state->ncols));
	state->backend = gftr_backend;
	state->mode = gftr_mode;
	state->device = gftr_device;
	state->device_bytes = (size_t) gftr_max_device_mb * 1024 * 1024;
	row_bytes = sizeof(uint64_t) + (uint64) state->ncols * sizeof(GftrValue);
	max_rows = (uint64) gftr_max_input_mb * 1024 * 1024 / row_bytes;
	/* Every palloc and CUB's item count must remain representable. */
	max_rows = Min(max_rows, (uint64) MaxAllocSize / sizeof(uint64_t));
	state->max_rows = (int) Min(max_rows, (uint64) INT_MAX);
	state->cleanup.func = gftr_cleanup;
	state->cleanup.arg = state;
	MemoryContextRegisterResetCallback(state->context, &state->cleanup);
	state->child = ExecInitNode((Plan *) linitial(plan->custom_plans), estate, eflags);
	node->custom_ps = list_make1(state->child);
}

static void
grow_input(GftrState *state)
{
	int capacity;
	int c;
	if (state->input.nrows >= state->max_rows)
		ereport(ERROR,
			(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
			 errmsg("pg_gftr packed input limit exceeded"),
			 errhint("Increase pg_gftr.max_input_mb or disable pg_gftr for this query.")));
	capacity = (int) Min((int64) state->max_rows,
						state->capacity ? (int64) state->capacity * 2 : 8192);
	if (!state->input.keys)
		state->input.keys = palloc((Size) capacity * sizeof(uint64_t));
	else
		state->input.keys = repalloc((void *) state->input.keys,
									(Size) capacity * sizeof(uint64_t));
	for (c = 0; c < state->ncols; c++)
	{
		if (!state->input.columns[c])
			state->input.columns[c] = palloc((Size) capacity * sizeof(GftrValue));
		else
			state->input.columns[c] = repalloc(state->input.columns[c],
											  (Size) capacity * sizeof(GftrValue));
	}
	state->capacity = capacity;
}

static void
load_input(GftrState *state)
{
	MemoryContext old = MemoryContextSwitchTo(state->context);
	TupleTableSlot *slot;
	instr_time start;
	instr_time end;
	char error[GFTR_ERROR_SIZE] = {0};
	int status;

	INSTR_TIME_SET_CURRENT(start);
	for (;;)
	{
		int row = state->input.nrows;
		bool isnull;
		Datum key;
		int c;
		/* TupIsNull is a macro: never pass an expression with side effects. */
		slot = ExecProcNode(state->child);
		if (TupIsNull(slot))
			break;
		CHECK_FOR_INTERRUPTS();
		if (row == state->capacity)
			grow_input(state);
		key = slot_getattr(slot, state->key_attno, &isnull);
		((uint64_t *) state->input.keys)[row] = isnull ? GFTR_NULL_KEY :
			(uint64_t) ((int64) DatumGetInt32(key) - PG_INT32_MIN);
		for (c = 0; c < state->ncols; c++)
		{
			Datum v = slot_getattr(slot, state->attnos[c], &isnull);
			state->input.columns[c][row].value = isnull ? 0 : DatumGetInt32(v);
			state->input.columns[c][row].valid = !isnull;
		}
		state->input.nrows++;
	}
	INSTR_TIME_SET_CURRENT(end);
	INSTR_TIME_SUBTRACT(end, start);
	state->pack_ms = INSTR_TIME_GET_MILLISEC(end);
	if (state->backend == 1)
		status = gftr_cpu_compute(&state->input, &state->result, error);
	else
		status = gftr_cuda_compute(&state->input, state->mode == 0,
			state->device, state->device_bytes, &state->result, error);
	CHECK_FOR_INTERRUPTS();
	if (status)
		ereport(ERROR, (errmsg("pg_gftr: %s", error)));
	state->loaded = true;
	MemoryContextSwitchTo(old);
}

static TupleTableSlot *
gftr_access(ScanState *scan)
{
	GftrState *state = (GftrState *) scan;
	TupleTableSlot *slot = scan->ss_ScanTupleSlot;
	int group;
	int out = 1;
	ListCell *lc;
	if (!state->loaded)
		load_input(state);
	ExecClearTuple(slot);
	if (state->next_group >= state->result.ngroups)
		return slot;
	group = state->next_group++;
	memset(slot->tts_isnull, 0, sizeof(bool) * slot->tts_tupleDescriptor->natts);
	if (state->result.keys[group] == GFTR_NULL_KEY)
		slot->tts_isnull[0] = true;
	else
		slot->tts_values[0] = Int32GetDatum((int32)
			((int64) state->result.keys[group] + PG_INT32_MIN));
	foreach(lc, state->ops)
	{
		List *op = lfirst(lc);
		int kind = linitial_int(op);
		int col = lsecond_int(op);
		GftrColumnStats *s;
		if (kind == GFTR_COUNT_STAR)
		{
			slot->tts_values[out++] = Int64GetDatum(state->result.counts[group]);
			continue;
		}
		s = &state->result.columns[col][group];
		if (kind == GFTR_COUNT)
			slot->tts_values[out] = Int64GetDatum(s->count);
		else if (s->count == 0)
			slot->tts_isnull[out] = true;
		else if (kind == GFTR_SUM)
			slot->tts_values[out] = Int64GetDatum(s->sum);
		else if (kind == GFTR_MIN)
			slot->tts_values[out] = Int32GetDatum(s->min);
		else
			slot->tts_values[out] = Int32GetDatum(s->max);
		out++;
	}
	return ExecStoreVirtualTuple(slot);
}

static bool
gftr_recheck(ScanState *state, TupleTableSlot *slot)
{
	return true;
}

static TupleTableSlot *
gftr_exec(CustomScanState *node)
{
	return ExecScan(&node->ss, gftr_access, gftr_recheck);
}

static void
gftr_end(CustomScanState *node)
{
	GftrState *state = (GftrState *) node;
	ExecEndNode(state->child);
	gftr_result_free(&state->result);
}

static void
gftr_rescan(CustomScanState *node)
{
	GftrState *state = (GftrState *) node;
	gftr_result_free(&state->result);
	state->loaded = false;
	state->input.nrows = 0;
	state->next_group = 0;
	state->pack_ms = 0;
	/* custom_ps children are not outerPlan/innerPlan; propagate params here. */
	if (node->ss.ps.chgParam != NULL)
		UpdateChangedParamSet(state->child, node->ss.ps.chgParam);
	if (state->child->chgParam == NULL)
		ExecReScan(state->child);
}

static void
gftr_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GftrState *state = (GftrState *) node;
	ExplainPropertyText("Backend", state->backend ? "CPU reference" : "CUDA", es);
	ExplainPropertyText("Algorithm", state->backend ? "CPU sort reference" :
		(state->mode ? "GFUR sort aggregation" : "GFTR sort aggregation"), es);
	ExplainPropertyInteger("Payload Columns", NULL, state->ncols, es);
	if (es->analyze && state->loaded)
	{
		ExplainPropertyText("Device", state->result.device_name, es);
		ExplainPropertyInteger("Input Rows", NULL, state->input.nrows, es);
		ExplainPropertyInteger("Groups", NULL, state->result.ngroups, es);
		ExplainPropertyFloat("Scan and Pack", "ms", state->pack_ms, 3, es);
		ExplainPropertyFloat("Backend Total", "ms", state->result.total_ms, 3, es);
		ExplainPropertyFloat("Sort", "ms", state->result.sort_ms, 3, es);
		ExplainPropertyFloat("Group and Reduce", "ms", state->result.reduce_ms, 3, es);
		ExplainPropertyFloat("Transfer", "ms", state->result.transfer_ms, 3, es);
	}
}

static Node *
fused_create_state(CustomScan *scan)
{
	GftrFusedAggregateState *state = palloc0(sizeof(*state));

	NodeSetTag(&state->css, T_CustomScanState);
	state->css.methods = &fused_exec_methods;
	return (Node *) state;
}

static void
fused_begin(CustomScanState *node, EState *estate, int eflags)
{
	GftrFusedAggregateState *state = (GftrFusedAggregateState *) node;
	CustomScan *plan = (CustomScan *) node->ss.ps.plan;
	List *attrs = lsecond(plan->custom_private);
	PlanState *child;

	state->group_attno = intVal(linitial(plan->custom_private));
	state->sum_attno = attrs == NIL ? 0 : linitial_int(attrs);
	state->ops = lthird(plan->custom_private);
	state->result_context = AllocSetContextCreate(estate->es_query_cxt,
		"GftrFusedJoinAggregate results", ALLOCSET_SMALL_SIZES);
	child = ExecInitNode((Plan *) linitial(plan->custom_plans), estate, eflags);
	if (!IsA(child, CustomScanState))
		elog(ERROR, "pg_gftr fused aggregate requires a custom join child");
	state->join = (CustomScanState *) child;
	pg_gftr_fused_prepare_join(state->join);
	node->custom_ps = list_make1(child);
}

static TupleTableSlot *
fused_access(ScanState *scan)
{
	GftrFusedAggregateState *state = (GftrFusedAggregateState *) scan;
	TupleTableSlot *slot = scan->ss_ScanTupleSlot;
	GftrFusedGroup *group;
	int out = 1;
	ListCell *lc;

	if (!state->loaded)
	{
		if (state->join->ss.ps.chgParam)
			ExecReScan(&state->join->ss.ps);
		pg_gftr_fused_reduce(state->join, state->group_attno, state->sum_attno,
							state->result_context, &state->result);
		state->loaded = true;
	}
	ExecClearTuple(slot);
	if (state->next_group >= state->result.ngroups)
		return slot;

	group = &state->result.groups[state->next_group++];
	slot->tts_isnull[0] = group->key_null;
	slot->tts_values[0] = Int32GetDatum(group->key);
	foreach(lc, state->ops)
	{
		int op = linitial_int(lfirst(lc));
		bool is_count = op == GFTR_COUNT_STAR;

		slot->tts_isnull[out] = !is_count && !group->sum_valid;
		slot->tts_values[out] = Int64GetDatum(is_count ? group->count : group->sum);
		out++;
	}
	return ExecStoreVirtualTuple(slot);
}

static TupleTableSlot *
fused_exec(CustomScanState *node)
{
	return ExecScan(&node->ss, fused_access, gftr_recheck);
}

static void
fused_end(CustomScanState *node)
{
	GftrFusedAggregateState *state = (GftrFusedAggregateState *) node;

	ExecEndNode(&state->join->ss.ps);
	MemoryContextDelete(state->result_context);
}

static void
fused_rescan(CustomScanState *node)
{
	GftrFusedAggregateState *state = (GftrFusedAggregateState *) node;
	PlanState *child = &state->join->ss.ps;

	ExecScanReScan(&node->ss);
	MemoryContextReset(state->result_context);
	memset(&state->result, 0, sizeof(state->result));
	state->loaded = false;
	state->next_group = 0;
	if (node->ss.ps.chgParam)
		UpdateChangedParamSet(child, node->ss.ps.chgParam);
	if (!child->chgParam)
		ExecReScan(child);
}

static void
fused_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
	GftrFusedAggregateState *state = (GftrFusedAggregateState *) node;

	ExplainPropertyText("Backend", "CPU", es);
	ExplainPropertyText("Algorithm", "Fused unique join and grouped reduction", es);
	ExplainPropertyBool("Partial", node->ss.ps.plan->parallel_safe, es);
	if (es->analyze && state->loaded)
	{
		if (node->ss.ps.plan->parallel_safe)
			ExplainPropertyText("Counter Scope", "Local participant", es);
		ExplainPropertyUInteger("Probe Rows", NULL, state->result.probe_rows, es);
		ExplainPropertyUInteger("Matched Rows", NULL, state->result.match_rows, es);
		ExplainPropertyInteger("Groups", NULL, state->result.ngroups, es);
		ExplainPropertyFloat("Build", "ms", state->result.build_ms, 3, es);
		ExplainPropertyFloat("Probe and Reduce", "ms", state->result.probe_ms, 3, es);
	}
}
