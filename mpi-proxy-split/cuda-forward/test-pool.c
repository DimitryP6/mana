// A test of pool.h and rbtree.h without a GPU: random allocations and frees
// on two devices, with host memory as chunks; after each step, the data, the
// red-black rules and the pool's bookkeeping are checked.
#include <stdio.h>
#include "pool.h"

#define CHUNK (2 << 20)
#define LIVE 600
#define OPS 100000

static uint64_t addr[LIVE];
static size_t size[LIVE];
static int dev[LIVE];
static unsigned char val[LIVE];
static int live;

static int
host_new(int d, size_t n, uint64_t *base, uint64_t *h)
{
  (void)d;
  void *m = malloc(n);
  *base = (uint64_t)(uintptr_t)m;
  *h = 0;
  return m == NULL;
}

static void
host_free(uint64_t base, size_t n, uint64_t h)
{
  (void)n;
  (void)h;
  free((void *)(uintptr_t)base);
}

static const RbNode *last;

// Black height of x's subtree, or -1 if a rule is broken.
static int
black_height(const RbTree *t, const RbNode *x)
{
  if (x == RB_NIL) {
    return 1;
  }
  if ((x->red && (x->l->red || x->r->red)) ||
      (x->l != RB_NIL && x->l->p != x) || (x->r != RB_NIL && x->r->p != x)) {
    return -1;
  }
  int a = black_height(t, x->l);
  if (a < 0 || (last != NULL && !t->less(last, x))) {   // in order
    return -1;
  }
  last = x;
  int b = black_height(t, x->r);
  return b < 0 || a != b ? -1 : a + !x->red;
}

static int
tree_ok(const RbTree *t)
{
  last = NULL;
  return !rb_nil.red && !t->root->red &&
         (t->root == RB_NIL || t->root->p == RB_NIL) &&
         black_height(t, t->root) > 0;
}

// Whether the blocks around 'b' are linked right and cover its chunk.
static int
links_ok(const PoolBlock *b)
{
  const PoolChunk *ch = b->ch;
  return (b->prev == NULL ? b->addr == ch->base
                          : b->prev->next == b &&
                            b->prev->addr + b->prev->size == b->addr) &&
         (b->next == NULL ? b->addr + b->size == ch->base + ch->size
                          : b->next->prev == b &&
                            b->addr + b->size == b->next->addr);
}

static int
pool_ok(Pool *p)
{
  if (!tree_ok(&p->free_tree) || !tree_ok(&p->used_tree) ||
      !tree_ok(&p->chunk_tree)) {
    return 0;
  }
  size_t free_bytes = 0, used_bytes = 0, chunk_bytes = 0;
  int n_used = 0;
  for (RbNode *n = rb_first(&p->free_tree); n != NULL; n = rb_next(n)) {
    PoolBlock *b = pool_block(n);
    if (b->used || !links_ok(b) || (b->next != NULL && !b->next->used)) {
      return 0;                   // used, broken links, or not merged
    }
    free_bytes += b->size;
    if (b->prev == NULL && b->next == NULL) {
      return 0;                   // an all-free chunk not given back
    }
  }
  for (PoolBlock *b = pool_first_used(p); b != NULL; b = pool_next_used(b)) {
    if (!b->used || !links_ok(b)) {
      return 0;
    }
    used_bytes += b->size;
    n_used++;
  }
  for (RbNode *n = rb_first(&p->chunk_tree); n != NULL; n = rb_next(n)) {
    chunk_bytes += pool_chunk(n)->size;
  }
  return free_bytes + used_bytes == chunk_bytes && chunk_bytes == p->held &&
         used_bytes == p->used && n_used == live;
}

static int
data_ok(int i)
{
  const unsigned char *m = (const unsigned char *)(uintptr_t)addr[i];
  for (size_t k = 0; k < size[i]; k += 509) {
    if (m[k] != val[i]) {
      return 0;
    }
  }
  return m[size[i] - 1] == val[i];
}

int
main(void)
{
  Pool p;
  pool_init(&p, CHUNK, host_new, host_free, -1);
  srand(3);
  for (int op = 0; op < OPS; op++) {
    int i = rand() % LIVE;
    if (addr[i] != 0) {
      if (!data_ok(i)) {
        printf("op %d: data of buffer %d overwritten\n", op, i);
        return 1;
      }
      if (rand() % 50 == 0 && pool_free(&p, addr[i] + 1) != 1) {
        printf("op %d: an address inside a block was not ignored\n", op);
        return 1;
      }
      if (pool_free(&p, addr[i]) != 1) {
        printf("op %d: free failed\n", op);
        return 1;
      }
      live--;
      if (rand() % 50 == 0) {
        pool_free(&p, addr[i]);   // a double free, to be ignored
      }
      addr[i] = 0;
    } else {
      int r = rand() % 100;
      size[i] = 512 * (size_t)(1 + (r < 60 ? rand() % 8 : rand() % 2048));
      dev[i] = rand() % 2;
      if (pool_alloc(&p, dev[i], size[i], &addr[i]) != 0) {
        printf("op %d: alloc of %zu failed\n", op, size[i]);
        return 1;
      }
      live++;
      if (pool_used_at(&p, addr[i])->ch->dev != dev[i] ||
          pool_block_size(&p, addr[i]) != size[i]) {
        printf("op %d: block on the wrong device or of the wrong size\n", op);
        return 1;
      }
      val[i] = (unsigned char)(op % 251 + 1);
      memset((void *)(uintptr_t)addr[i], val[i], size[i]);
    }
    if (!pool_ok(&p)) {
      printf("op %d: trees or bookkeeping broken\n", op);
      return 1;
    }
  }
  int x;
  if (pool_free(&p, (uint64_t)(uintptr_t)&x) != 0 ||
      pool_alloc(&p, 0, CHUNK + 512, &addr[0]) == 0) {
    printf("a foreign address or an oversized request was accepted\n");
    return 1;
  }
  for (int i = 0; i < LIVE; i++) {
    if (addr[i] != 0) {
      pool_free(&p, addr[i]);
      live--;
    }
  }
  if (!pool_ok(&p) || p.used != 0 || p.held != 0) {
    printf("after freeing all: used %zu, held %zu (want 0 and 0)\n", p.used,
           p.held);
    return 1;
  }
  printf("PASS: %d operations; peak %zu bytes used in %zu of chunks\n", OPS,
         p.used_max, p.held_max);
  return 0;
}