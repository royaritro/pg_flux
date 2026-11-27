#ifndef GFTR_JOIN_H
#define GFTR_JOIN_H

#include "gftr_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Build keys are non-NULL built-in int4 values. Probe NULLs use validity bytes.
 * Probe results are build row indices, or -1 for a nonmatch, in probe order.
 * Calls return zero on success. Destructors accept NULL. */
typedef struct GftrJoinBuild GftrJoinBuild;

typedef struct GftrJoinTiming
{
	double build_ms;
	double h2d_ms;
	double probe_ms;
	double d2h_ms;
	size_t build_bytes;
	size_t h2d_bytes;
	size_t d2h_bytes;
	const char *algorithm;
} GftrJoinTiming;

int gftr_join_cpu_build(const int32_t *keys, int32_t count, size_t budget,
						GftrJoinBuild **out, GftrJoinTiming *timing,
						char error[GFTR_ERROR_SIZE]);
/* Build a lookup of nonnegative group indices instead of build row indices. */
int gftr_join_cpu_build_groups(const int32_t *keys, const int32_t *groups,
							   int32_t count, size_t budget, GftrJoinBuild **out,
							   GftrJoinTiming *timing, char error[GFTR_ERROR_SIZE]);
/* Return -1 when the key has no match. */
int32_t gftr_join_cpu_lookup(const GftrJoinBuild *build, int32_t key);
int gftr_join_cpu_probe(GftrJoinBuild *build, const int32_t *keys,
						const uint8_t *valid, int32_t count, int32_t *matches,
						GftrJoinTiming *timing, char error[GFTR_ERROR_SIZE]);
void gftr_join_cpu_free(GftrJoinBuild *build);

int gftr_join_cuda_build(const int32_t *keys, int32_t count, int device,
						 int32_t batch_rows, size_t budget,
						 GftrJoinBuild **out, GftrJoinTiming *timing,
						 char error[GFTR_ERROR_SIZE]);
int gftr_join_cuda_probe(GftrJoinBuild *build, const int32_t *keys,
						 const uint8_t *valid, int32_t count, int32_t *matches,
						 GftrJoinTiming *timing, char error[GFTR_ERROR_SIZE]);
void gftr_join_cuda_free(GftrJoinBuild *build);

void gftr_join_init(void);
void gftr_join_fini(void);

#ifdef __cplusplus
}
#endif

#endif
