#ifndef GFTR_JOIN_LOOKUP_H
#define GFTR_JOIN_LOOKUP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __CUDACC__
#define GFTR_JOIN_INLINE static inline __host__ __device__
#else
#define GFTR_JOIN_INLINE static inline
#endif

/* A hash slot packs the signed key's bits and the build row index. An
 * occupied slot cannot have all bits set because row indices are nonnegative. */
typedef unsigned long long GftrJoinEntry;
#define GFTR_JOIN_EMPTY (~(GftrJoinEntry) 0)

typedef struct GftrJoinLayout
{
	int32_t minimum;
	size_t slots;
	size_t bytes;
	int dense;
} GftrJoinLayout;

static inline int
gftr_join_layout(const int32_t *keys, int32_t count, GftrJoinLayout *layout)
{
	int32_t minimum;
	int32_t maximum;
	int32_t i;
	uint64_t range;
	uint64_t hash_slots = 1;
	uint64_t slots;
	size_t slot_bytes;

	*layout = (GftrJoinLayout) {0, 0, 0, 1};
	if (count == 0)
		return 0;
	minimum = maximum = keys[0];
	for (i = 1; i < count; i++)
	{
		if (keys[i] < minimum)
			minimum = keys[i];
		if (keys[i] > maximum)
			maximum = keys[i];
	}

	range = (uint64_t) ((int64_t) maximum - minimum) + 1;
	while (hash_slots < (uint64_t) count * 2)
		hash_slots *= 2;

	/* Prefer one access and four-byte entries when the dense table uses
	 * at most as much memory as the half-full, power-of-two hash table. */
	layout->dense = range <= hash_slots * 2;
	slots = layout->dense ? range : hash_slots;
	slot_bytes = layout->dense ? sizeof(int32_t) : sizeof(GftrJoinEntry);
	if (slots > SIZE_MAX / slot_bytes)
		return -1;
	layout->minimum = minimum;
	layout->slots = (size_t) slots;
	layout->bytes = (size_t) slots * slot_bytes;
	return 0;
}

GFTR_JOIN_INLINE size_t
gftr_join_hash(int32_t key, size_t mask)
{
	uint32_t word = (uint32_t) key;

	word ^= word >> 16;
	word *= UINT32_C(0x7feb352d);
	word ^= word >> 15;
	word *= UINT32_C(0x846ca68b);
	word ^= word >> 16;
	return (size_t) word & mask;
}

GFTR_JOIN_INLINE GftrJoinEntry
gftr_join_entry(int32_t key, int32_t index)
{
	return ((GftrJoinEntry) (uint32_t) key << 32) | (uint32_t) index;
}

GFTR_JOIN_INLINE int
gftr_join_is_key(GftrJoinEntry entry, int32_t key)
{
	return (uint32_t) (entry >> 32) == (uint32_t) key;
}

GFTR_JOIN_INLINE int32_t
gftr_join_dense_lookup(const int32_t *table, GftrJoinLayout layout, int32_t key)
{
	uint64_t offset = (uint64_t) ((int64_t) key - layout.minimum);

	return offset < layout.slots ? table[offset] : -1;
}

GFTR_JOIN_INLINE int32_t
gftr_join_hash_lookup(const GftrJoinEntry *table, size_t slots, int32_t key)
{
	size_t mask = slots - 1;
	size_t slot = gftr_join_hash(key, mask);
	GftrJoinEntry entry = table[slot];

	while (entry != GFTR_JOIN_EMPTY)
	{
		if (gftr_join_is_key(entry, key))
			return (int32_t) (uint32_t) entry;
		slot = (slot + 1) & mask;
		entry = table[slot];
	}
	return -1;
}

#undef GFTR_JOIN_INLINE
#endif
