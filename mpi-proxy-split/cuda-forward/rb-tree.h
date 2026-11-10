#ifndef MANA_RBTREE_H
#define MANA_RBTREE_H

#include <stddef.h>

typedef struct RbNode {
  struct RbNode *l, *r, *p;
  int red;
} RbNode;

typedef struct {
  RbNode *root;
  int (*less)(const RbNode *, const RbNode *);
} RbTree;

#define RB_ENTRY(node, type, member) \
  ((type *)((char *)(node) - offsetof(type, member)))

// Black leaf.
static RbNode rb_nil __attribute__((unused)) = { &rb_nil, &rb_nil, &rb_nil, 0 };
#define RB_NIL (&rb_nil)

static inline void
rb_init(RbTree *t, int (*less)(const RbNode *, const RbNode *))
{
  t->root = RB_NIL;
  t->less = less;
}

static inline void
rb_rotate_left(RbTree *t, RbNode *x)
{
  RbNode *y = x->r;
  x->r = y->l;
  if (y->l != RB_NIL) {
    y->l->p = x;
  }
  y->p = x->p;
  if (x->p == RB_NIL) {
    t->root = y;
  } else if (x == x->p->l) {
    x->p->l = y;
  } else {
    x->p->r = y;
  }
  y->l = x;
  x->p = y;
}

static inline void
rb_rotate_right(RbTree *t, RbNode *x)
{
  RbNode *y = x->l;
  x->l = y->r;
  if (y->r != RB_NIL) {
    y->r->p = x;
  }
  y->p = x->p;
  if (x->p == RB_NIL) {
    t->root = y;
  } else if (x == x->p->r) {
    x->p->r = y;
  } else {
    x->p->l = y;
  }
  y->r = x;
  x->p = y;
}

static inline void
rb_insert(RbTree *t, RbNode *z)
{
  RbNode *y = RB_NIL, *x = t->root;
  while (x != RB_NIL) {
    y = x;
    x = t->less(z, x) ? x->l : x->r;
  }
  z->p = y;
  if (y == RB_NIL) {
    t->root = z;
  } else if (t->less(z, y)) {
    y->l = z;
  } else {
    y->r = z;
  }
  z->l = z->r = RB_NIL;
  z->red = 1;
  while (z->p->red) {
    RbNode *g = z->p->p;
    if (z->p == g->l) {
      RbNode *u = g->r;
      if (u->red) {
        z->p->red = 0;
        u->red = 0;
        g->red = 1;
        z = g;
      } else {
        if (z == z->p->r) {
          z = z->p;
          rb_rotate_left(t, z);
        }
        z->p->red = 0;
        g->red = 1;
        rb_rotate_right(t, g);
      }
    } else {
      RbNode *u = g->l;
      if (u->red) {
        z->p->red = 0;
        u->red = 0;
        g->red = 1;
        z = g;
      } else {
        if (z == z->p->l) {
          z = z->p;
          rb_rotate_right(t, z);
        }
        z->p->red = 0;
        g->red = 1;
        rb_rotate_left(t, g);
      }
    }
  }
  t->root->red = 0;
}

static inline void
rb_transplant(RbTree *t, RbNode *u, RbNode *v)
{
  if (u->p == RB_NIL) {
    t->root = v;
  } else if (u == u->p->l) {
    u->p->l = v;
  } else {
    u->p->r = v;
  }
  v->p = u->p;
}

// Restore: every path from a node to a leaf has as many black nodes.
static inline void
rb_remove_fixup(RbTree *t, RbNode *x)
{
  while (x != t->root && !x->red) {
    if (x == x->p->l) {
      RbNode *w = x->p->r;
      if (w->red) {
        w->red = 0;
        x->p->red = 1;
        rb_rotate_left(t, x->p);
        w = x->p->r;
      }
      if (!w->l->red && !w->r->red) {
        w->red = 1;
        x = x->p;
      } else {
        if (!w->r->red) {
          w->l->red = 0;
          w->red = 1;
          rb_rotate_right(t, w);
          w = x->p->r;
        }
        w->red = x->p->red;
        x->p->red = 0;
        w->r->red = 0;
        rb_rotate_left(t, x->p);
        x = t->root;
      }
    } else {
      RbNode *w = x->p->l;
      if (w->red) {
        w->red = 0;
        x->p->red = 1;
        rb_rotate_right(t, x->p);
        w = x->p->l;
      }
      if (!w->r->red && !w->l->red) {
        w->red = 1;
        x = x->p;
      } else {
        if (!w->l->red) {
          w->r->red = 0;
          w->red = 1;
          rb_rotate_left(t, w);
          w = x->p->l;
        }
        w->red = x->p->red;
        x->p->red = 0;
        w->l->red = 0;
        rb_rotate_right(t, x->p);
        x = t->root;
      }
    }
  }
  x->red = 0;
}

static inline void
rb_remove(RbTree *t, RbNode *z)
{
  RbNode *y = z, *x;
  int y_red = y->red;
  if (z->l == RB_NIL) {
    x = z->r;
    rb_transplant(t, z, z->r);
  } else if (z->r == RB_NIL) {
    x = z->l;
    rb_transplant(t, z, z->l);
  } else {
    for (y = z->r; y->l != RB_NIL; y = y->l) {   // z's successor
    }
    y_red = y->red;
    x = y->r;
    if (y->p == z) {
      x->p = y;
    } else {
      rb_transplant(t, y, y->r);
      y->r = z->r;
      y->r->p = y;
    }
    rb_transplant(t, z, y);
    y->l = z->l;
    y->l->p = y;
    y->red = z->red;
  }
  if (!y_red) {
    rb_remove_fixup(t, x);
  }
}

// The first node in order for which before(node, key) is false, or NULL:
// the smallest node >= key, if 'before' tells node < key.
static inline RbNode *
rb_lower_bound(const RbTree *t, const void *key,
               int (*before)(const RbNode *, const void *))
{
  RbNode *best = NULL;
  for (RbNode *x = t->root; x != RB_NIL;) {
    if (before(x, key)) {
      x = x->r;
    } else {
      best = x;
      x = x->l;
    }
  }
  return best;
}

// The last node in order for which after(node, key) is false, or NULL: the
// largest node <= key, if 'after' tells node > key.
static inline RbNode *
rb_floor(const RbTree *t, const void *key,
         int (*after)(const RbNode *, const void *))
{
  RbNode *best = NULL;
  for (RbNode *x = t->root; x != RB_NIL;) {
    if (after(x, key)) {
      x = x->l;
    } else {
      best = x;
      x = x->r;
    }
  }
  return best;
}

// In-order iteration: for (n = rb_first(t); n != NULL; n = rb_next
static inline RbNode *
rb_first(const RbTree *t)
{
  RbNode *x = t->root;
  if (x == RB_NIL) {
    return NULL;
  }
  while (x->l != RB_NIL) {
    x = x->l;
  }
  return x;
}

static inline RbNode *
rb_next(RbNode *x)
{
  if (x->r != RB_NIL) {
    for (x = x->r; x->l != RB_NIL; x = x->l) {
    }
    return x;
  }
  RbNode *y = x->p;
  while (y != RB_NIL && x == y->r) {
    x = y;
    y = y->p;
  }
  return y == RB_NIL ? NULL : y;
}

#endif