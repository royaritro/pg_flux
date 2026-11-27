#include "gftr_join.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct JoinBackend
{
	int cuda;
	GftrJoinBuild *state;
	GftrJoinTiming timing;
	char error[GFTR_ERROR_SIZE];
} JoinBackend;

static int
build(JoinBackend *backend, const int32_t *keys, int32_t count, size_t budget)
{
	if (backend->cuda)
		return gftr_join_cuda_build(keys, count, 0, 4, budget, &backend->state,
									&backend->timing, backend->error);
	return gftr_join_cpu_build(keys, count, budget, &backend->state,
							   &backend->timing, backend->error);
}

static int
probe(JoinBackend *backend, const int32_t *keys, const uint8_t *valid,
	  int32_t count, int32_t *matches)
{
	if (backend->cuda)
		return gftr_join_cuda_probe(backend->state, keys, valid, count, matches,
									&backend->timing, backend->error);
	return gftr_join_cpu_probe(backend->state, keys, valid, count, matches,
							   &backend->timing, backend->error);
}

static void
release(JoinBackend *backend)
{
	if (backend->cuda)
		gftr_join_cuda_free(backend->state);
	else
		gftr_join_cpu_free(backend->state);
	backend->state = NULL;
}

static void
check_matches(JoinBackend *backend, const int32_t *keys, const uint8_t *valid,
			  const int32_t *expected, int32_t count)
{
	int32_t matches[4];

	assert(probe(backend, keys, valid, count, matches) == 0);
	assert(memcmp(matches, expected, (size_t) count * sizeof(*matches)) == 0);
}

static void
test_lookup(int use_cuda)
{
	JoinBackend backend = {.cuda = use_cuda};
	const int32_t dense_keys[] = {2, -1, 0, -2};
	const int32_t dense_probe[] = {-1, 1, -2, 2};
	const int32_t dense_expected[] = {1, -1, -1, 0};
	const uint8_t valid[] = {1, 1, 0, 1};
	const int32_t dense_again[] = {2, 0, 2, INT32_MAX};
	const int32_t dense_again_expected[] = {0, 2, 0, -1};
	const int32_t sparse_keys[] = {INT32_MAX, -1, INT32_MIN, 1024};
	const int32_t sparse_probe[] = {INT32_MIN, INT32_MAX, -1, INT32_MIN + 1};
	const int32_t sparse_expected[] = {2, 0, 1, -1};
	const int32_t no_matches[] = {-1, -1, -1, -1};
	const int32_t dense_duplicates[] = {-1, -1};
	const int32_t sparse_duplicates[] = {-1, INT32_MIN, -1};

	assert(build(&backend, dense_keys, 4, 4096) == 0);
	assert(strcmp(backend.timing.algorithm, "dense") == 0);
	assert(backend.timing.build_bytes == 5 * sizeof(int32_t));
	assert(probe(&backend, NULL, NULL, 0, NULL) == 0);
	check_matches(&backend, dense_probe, valid, dense_expected, 4);
	check_matches(&backend, dense_again, NULL, dense_again_expected, 4);
	if (use_cuda)
	{
		assert(probe(&backend, dense_probe, valid, 5, NULL) != 0);
		assert(backend.timing.h2d_bytes == 12 * sizeof(int32_t) + sizeof(valid));
		assert(backend.timing.d2h_bytes == 8 * sizeof(int32_t) + sizeof(int));
	}
	release(&backend);

	assert(build(&backend, sparse_keys, 4, 4096) == 0);
	assert(strcmp(backend.timing.algorithm, "hash") == 0);
	check_matches(&backend, sparse_probe, NULL, sparse_expected, 4);
	release(&backend);

	assert(build(&backend, NULL, 0, 0) == 0);
	assert(strcmp(backend.timing.algorithm, "empty") == 0);
	check_matches(&backend, dense_probe, valid, no_matches, 4);
	release(&backend);

	assert(build(&backend, dense_keys, 4, 1) != 0);
	assert(backend.state == NULL && strstr(backend.error, "budget"));
	assert(build(&backend, dense_duplicates, 2, 4096) != 0);
	assert(backend.state == NULL && strstr(backend.error, "duplicate"));
	assert(build(&backend, sparse_duplicates, 3, 4096) != 0);
	assert(backend.state == NULL && strstr(backend.error, "duplicate"));
	release(&backend);
}

int
main(int argc, char **argv)
{
	test_lookup(0);
	puts("CPU join backend checks passed");
	if (argc > 1 && strcmp(argv[1], "--cuda") == 0)
	{
		test_lookup(1);
		puts("CUDA join backend checks passed");
	}
	return 0;
}
