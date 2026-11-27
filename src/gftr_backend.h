#ifndef GFTR_BACKEND_H
#define GFTR_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Signed int4 keys map to [0, 2^32); SQL NULL sorts last. */
#define GFTR_NULL_KEY (UINT64_C(1) << 32)
#define GFTR_ERROR_SIZE 512

typedef struct GftrValue
{
	int32_t value; // The actual value of the column for this row (0 if NULL)
	uint32_t valid; // Flag indicating if the value is valid (not NULL) 1 if valid, 0 if NULL
} GftrValue;

typedef struct GftrColumnStats
{
	int64_t count;
	int64_t sum;
	int32_t min;
	int32_t max;
} GftrColumnStats;

/**
 * GftrInput is everything needed to perform the aggregation: the number of rows, the number of columns, the keys, and the column values.
 * 
 * Example:
 * SELECT region, sum(price), max(qty) FROM sales GROUP BY region;
 *  row | region | price | qty
   -----+--------+-------+-----
	 0  |    2   |  100  |  5
	 1  |   -1   |  NULL |  3
	 2  |    2   |  250  |  7
	 3  |  NULL  |   40  |  1
	 4  |   -1   |   60  | NULL

 * From the planner: key_attno points at region, attnos = [price, qty], so ncols = 2. 
 nrows = 5
 ncols = 2

 keys (uint64, key - INT32_MIN; NULL → 1<<32):
  [0] 2147483650     ← region 2
  [1] 2147483647     ← region -1
  [2] 2147483650     ← region 2
  [3] 4294967296     ← region NULL  (GFTR_NULL_KEY)
  [4] 2147483647     ← region -1

  columns[0]  (price, GftrValue {value, valid}):
  [0] {100, 1}
  [1] {  0, 0}       ← NULL
  [2] {250, 1}
  [3] { 40, 1}
  [4] { 60, 1}

columns[1]  (qty):
  [0] {5, 1}
  [1] {3, 1}
  [2] {7, 1}
  [3] {1, 1}
  [4] {0, 0}         ← NULL
 */
typedef struct GftrInput
{
	int32_t nrows; // how many rows were packed
	int32_t ncols; // how many value columns (== state->ncols)
	const uint64_t *keys; // keys[row]: GROUP BY key for each row
	GftrValue **columns; // columns[col][row]: value for each row in each column
} GftrInput;

typedef struct GftrResult
{
	int32_t ngroups;
	int32_t ncols;
	uint64_t *keys;
	int32_t *counts;
	GftrColumnStats **columns;
	double sort_ms;
	double reduce_ms;
	double transfer_ms;
	double total_ms;
	char device_name[128];
} GftrResult;

int gftr_result_alloc(GftrResult *result, int32_t capacity, int32_t ncols);
void gftr_result_free(GftrResult *result);
int gftr_cpu_compute(const GftrInput *input, GftrResult *result,
					 char error[GFTR_ERROR_SIZE]);
int gftr_cuda_compute(const GftrInput *input, int use_gftr, int device,
					  size_t max_device_bytes, GftrResult *result,
					  char error[GFTR_ERROR_SIZE]);

#ifdef __cplusplus
}
#endif
#endif
