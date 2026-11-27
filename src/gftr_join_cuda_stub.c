#include "gftr_join.h"

#include <stdio.h>

int
gftr_join_cuda_build(const int32_t *keys, int32_t count, int device,
					 int32_t batch_rows, size_t budget, GftrJoinBuild **out,
					 GftrJoinTiming *timing, char error[GFTR_ERROR_SIZE])
{
	(void) keys;
	(void) count;
	(void) device;
	(void) batch_rows;
	(void) budget;
	(void) timing;
	*out = NULL;
	snprintf(error, GFTR_ERROR_SIZE, "CUDA join support was not built; rebuild with USE_CUDA=1");
	return -1;
}

int
gftr_join_cuda_probe(GftrJoinBuild *build, const int32_t *keys,
					 const uint8_t *valid, int32_t count, int32_t *matches,
					 GftrJoinTiming *timing, char error[GFTR_ERROR_SIZE])
{
	(void) build;
	(void) keys;
	(void) valid;
	(void) count;
	(void) matches;
	(void) timing;
	snprintf(error, GFTR_ERROR_SIZE, "CUDA join support was not built; rebuild with USE_CUDA=1");
	return -1;
}

void
gftr_join_cuda_free(GftrJoinBuild *build)
{
	(void) build;
}
