#ifndef MANA_POOL_H
#define MANA_POOL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "rb-tree.h"

typedef struct PoolChunk {
  uint64_t base;
  size_t size;
  uint64_t h;                     // chunk_new()'s, for chunk_free()
  int dev;
  RbNode rb;                      // in chunk_tree, by base
} PoolChunk;

// A piece of a chunk, free or used.  The blocks of a chunk cover it, linked by
// address.
typedef struct PoolBlock {
  uint64_t addr;
  size_t size;
  int used;
  PoolChunk *ch;
  struct PoolBlock *prev, *next;  // the neighbors in the chunk
  RbNode rb;                      // in free_tree if free, used_tree if used
} PoolBlock;

typedef struct Pool {
  size_t chunk_size;
  // A new chunk of 'size' bytes on 'dev': its address in *base and a value for
  // chunk_free() in *h.  0, or an error, which pool_alloc() returns.
  int (*chunk_new)(int dev, size_t size, uint64_t *base, uint64_t *h);
  void (*chunk_free)(uint64_t base, size_t size, uint64_t h);
  int nomem;                      // pool_alloc()'s error without host memory
  int ready;                      // pool_init() was called
  RbTree free_tree;               // free blocks by device, size, address
  RbTree used_tree;               // used blocks by address
  RbTree chunk_tree;              // chunks by base
  size_t used, held;              // bytes of used blocks and of chunks
  size_t used_max, held_max;      // their peaks
} Pool;

static inline PoolBlock *
pool_block(const RbNode *n)
{
  return RB_ENTRY(n, PoolBlock, rb);
}

static inline PoolChunk *
pool_chunk(const RbNode *n)
{
  return RB_ENTRY(n, PoolChunk, rb);
}

static inline int
pool_free_less(const RbNode *a, const RbNode *b)
{
  const PoolBlock *x = pool_block(a), *y = pool_block(b);
  if (x->ch->dev != y->ch->dev) {
    return x->ch->dev < y->ch->dev;
  }
  if (x->size != y->size) {
    return x->size < y->size;
  }
  return x->addr < y->addr;
}

static inline int
pool_used_less(const RbNode *a, const RbNode *b)
{
  return pool_block(a)->addr < pool_block(b)->addr;
}

static inline int
pool_chunk_less(const RbNode *a, const RbNode *b)
{
  return pool_chunk(a)->base < pool_chunk(b)->base;
}

struct pool_fit {
  int dev;
  size_t size;
};

// Whether free block 'n' comes before every block that fits 'key'.
static inline int
pool_fit_before(const RbNode *n, const void *key)
{
  const PoolBlock *b = pool_block(n);
  const struct pool_fit *k = ((const struct pool_fit*)key);
  return b->ch->dev < k->dev || (b->ch->dev == k->dev && b->size < k->size);
}

static inline int
pool_block_after(const RbNode *n, const void *addr)
{
  return pool_block(n)->addr > *(const uint64_t *)addr;
}

static inline int
pool_chunk_after(const RbNode *n, const void *addr)
{
  return pool_chunk(n)->base > *(const uint64_t *)addr;
}

static inline void
pool_init(Pool *p, size_t chunk_size,
          int (*chunk_new)(int, size_t, uint64_t *, uint64_t *),
          void (*chunk_free)(uint64_t, size_t, uint64_t), int nomem)
{
  memset(p, 0, sizeof *p);
  p->chunk_size = chunk_size;
  p->chunk_new = chunk_new;
  p->chunk_free = chunk_free;
  p->nomem = nomem;
  rb_init(&p->free_tree, pool_free_less);
  rb_init(&p->used_tree, pool_used_less);
  rb_init(&p->chunk_tree, pool_chunk_less);
  p->ready = 1;
}

static inline void
pool_count(size_t *now, size_t *max, size_t n)
{
  *now += n;
  if (*now > *max) {
    *max = *now;
  }
}

// A new chunk on 'dev'; *bp gets its one block, free but in no tree.
static inline int
pool_chunk_new(Pool *p, int dev, PoolBlock **bp)
{
  PoolChunk *ch = ((PoolChunk*)malloc(sizeof *ch));
  PoolBlock *b = ((PoolBlock*)malloc(sizeof *b));
  if (ch == NULL || b == NULL) {
    free(ch);
    free(b);
    return p->nomem;
  }
  int rc = p->chunk_new(dev, p->chunk_size, &ch->base, &ch->h);
  if (rc != 0) {
    free(ch);
    free(b);
    return rc;
  }
  ch->size = p->chunk_size;
  ch->dev = dev;
  rb_insert(&p->chunk_tree, &ch->rb);
  pool_count(&p->held, &p->held_max, ch->size);
  *b = (PoolBlock){ .addr = ch->base, .size = ch->size, .ch = ch };
  *bp = b;
  return 0;
}

static inline void
pool_chunk_free(Pool *p, PoolChunk *ch)
{
  p->chunk_free(ch->base, ch->size, ch->h);
  rb_remove(&p->chunk_tree, &ch->rb);
  p->held -= ch->size;
  free(ch);
}

// A block of 'size' bytes on 'dev', best fit: its address in *addr.  0, or
// chunk_new()'s error, or nomem.
static inline int
pool_alloc(Pool *p, int dev, size_t size, uint64_t *addr)
{
  if (!p->ready || size == 0 || size > p->chunk_size) {
    return p->nomem;
  }
  struct pool_fit key = { dev, size };
  RbNode *n = rb_lower_bound(&p->free_tree, &key, pool_fit_before);
  PoolBlock *b = n != NULL && pool_block(n)->ch->dev == dev ? pool_block(n)
                                                            : NULL;
  if (b != NULL) {
    rb_remove(&p->free_tree, &b->rb);   // before its size changes
  } else {
    int rc = pool_chunk_new(p, dev, &b);
    if (rc != 0) {
      return rc;
    }
  }
  // Split off the rest, unless there is no memory for its block: then the
  // whole block.
  PoolBlock *rest = (size_t)b->size > size ? ((PoolBlock*)malloc(sizeof *rest)) : NULL;
  if (rest != NULL) {
    *rest = (PoolBlock){ .addr = b->addr + size, .size = b->size - size,
                         .ch = b->ch, .prev = b, .next = b->next };
    if (b->next != NULL) {
      b->next->prev = rest;
    }
    b->next = rest;
    b->size = size;
    rb_insert(&p->free_tree, &rest->rb);
  }
  b->used = 1;
  rb_insert(&p->used_tree, &b->rb);
  pool_count(&p->used, &p->used_max, b->size);
  *addr = b->addr;
  return 0;
}

// The used block that contains 'addr', or NULL.
static inline PoolBlock *
pool_used_at(const Pool *p, uint64_t addr)
{
  if (!p->ready) {
    return NULL;
  }
  RbNode *n = rb_floor(&p->used_tree, &addr, pool_block_after);
  PoolBlock *b = n != NULL ? pool_block(n) : NULL;
  return b != NULL && addr - b->addr < b->size ? b : NULL;
}

// Whether 'addr' is in a chunk.
static inline int
pool_in_chunk(const Pool *p, uint64_t addr)
{
  if (!p->ready) {
    return 0;
  }
  RbNode *n = rb_floor(&p->chunk_tree, &addr, pool_chunk_after);
  return n != NULL && addr - pool_chunk(n)->base < pool_chunk(n)->size;
}

// The size of the used block that starts at 'addr', or 0.
static inline size_t
pool_block_size(const Pool *p, uint64_t addr)
{
  PoolBlock *b = pool_used_at(p, addr);
  return b != NULL && b->addr == addr ? b->size : 0;
}

// The used blocks by address:
//   for (b = pool_first_used(p); b != NULL; b = pool_next_used(b))
static inline PoolBlock *
pool_first_used(const Pool *p)
{
  RbNode *n = p->ready ? rb_first(&p->used_tree) : NULL;
  return n != NULL ? pool_block(n) : NULL;
}

static inline PoolBlock *
pool_next_used(PoolBlock *b)
{
  RbNode *n = rb_next(&b->rb);
  return n != NULL ? pool_block(n) : NULL;
}

// Frees the used block that starts at 'addr'.  1 if 'addr' is in a chunk
// (an address that is no used block's start is ignored, as by CUDA), 0 if not.
static inline int
pool_free(Pool *p, uint64_t addr)
{
  PoolBlock *b = pool_used_at(p, addr);
  if (b == NULL || b->addr != addr) {
    return b != NULL || pool_in_chunk(p, addr);
  }
  rb_remove(&p->used_tree, &b->rb);
  b->used = 0;
  p->used -= b->size;
  // Merge with free neighbors, each out of free_tree first.
  PoolBlock *n = b->next;
  if (n != NULL && !n->used) {
    rb_remove(&p->free_tree, &n->rb);
    b->size += n->size;
    b->next = n->next;
    if (n->next != NULL) {
      n->next->prev = b;
    }
    free(n);
  }
  PoolBlock *q = b->prev;
  if (q != NULL && !q->used) {
    rb_remove(&p->free_tree, &q->rb);
    q->size += b->size;
    q->next = b->next;
    if (b->next != NULL) {
      b->next->prev = q;
    }
    free(b);
    b = q;
  }
  if (b->prev == NULL && b->next == NULL) {   // the chunk is all free
    pool_chunk_free(p, b->ch);
    free(b);
    return 1;
  }
  rb_insert(&p->free_tree, &b->rb);
  return 1;
}

#endif