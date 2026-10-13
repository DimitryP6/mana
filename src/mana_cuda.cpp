/****************************************************************************
 *   Copyright (C) 2026 by Yao Xu                                           *
 *   xu.yao1@northeastern.edu                                               *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License in the files COPYING and COPYING.LESSER.  If not, see           *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>

#include "mana_cuda.h"
#include "cuda-forward/fwd-runtime.h"
#include "dmtcp.h"
#include "lower-half-api.h"
#include "switch-context.h"
#include "uh_wrappers.h"

extern "C" pid_t dmtcp_get_real_pid();
extern "C" pid_t dmtcp_get_real_tid() __attribute((weak));

// From cuda.h, which MANA does not need to build.
enum {
  CU_POINTER_ATTRIBUTE_CONTEXT = 1,
  CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
  CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
  CU_MEMORYTYPE_DEVICE = 2,
};

static FwdCtl *
fwd_ctl()
{
  return lh_info != NULL ? (FwdCtl *)lh_info->fwd_ctl : NULL;
}

bool
mana_cuda_enabled()
{
  return fwd_ctl() != NULL;
}

// For the shims (FwdCtl::uh_dlsym): 'symbol' of library 'soname' of the
// upper half, which is loaded if it is not yet.
static void *
uh_dlsym(const char *soname, const char *symbol)
{
  void *handle = dlopen(soname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  if (handle == NULL) {
    handle = dlopen(soname, RTLD_NOW | RTLD_GLOBAL);
  }
  return handle != NULL ? dlsym(handle, symbol) : NULL;
}

void
mana_lower_half_mpi_init()
{
  initialize_wrappers();
  if (lh_info->lh_mpi_init != NULL) {
    mana_cuda_note_thread(dmtcp_get_real_tid(), getFS());
    JUMP_TO_LOWER_HALF(lh_info->fsaddr);
    ((void (*)(void))lh_info->lh_mpi_init)();
    RETURN_TO_UPPER_HALF();
  }
}

void
mana_cuda_note_thread(pid_t real_tid, unsigned long fs)
{
  initialize_wrappers();
  FwdCtl *ctl = fwd_ctl();
  if (ctl == NULL) {
    return;
  }
  ctl->uh_dlsym = uh_dlsym;
  unsigned long lo = 0, hi = 0;
  if (real_tid == dmtcp_get_real_pid()) {
    // The main thread's stack is the one the lower half made.
    lo = (unsigned long)lh_info->uh_stack_start;
    hi = (unsigned long)lh_info->uh_stack_end;
  } else {
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) == 0) {
      void *addr;
      size_t size;
      if (pthread_attr_getstack(&attr, &addr, &size) == 0) {
        lo = (unsigned long)addr;
        hi = lo + size;
      }
      pthread_attr_destroy(&attr);
    }
  }
  ctl->register_thread(real_tid, fs, lo, hi);
}

void
mana_cuda_forget_thread(pid_t real_tid)
{
  FwdCtl *ctl = fwd_ctl();
  if (ctl != NULL) {
    ctl->forget_thread(real_tid);
  }
}

// FwdCtl's checkpoint hooks are lower-half code that uses the lower half's
// libc: they run with its FS.
static void
call_in_lower_half(void (*fn)(void))
{
  JUMP_TO_LOWER_HALF(lh_info->fsaddr);
  fn();
  RETURN_TO_UPPER_HALF();
}

void
mana_cuda_before_ckpt()
{
  if (mana_cuda_enabled()) {
    call_in_lower_half(fwd_ctl()->before_ckpt);
  }
}

void
mana_cuda_after_resume()
{
  if (mana_cuda_enabled()) {
    call_in_lower_half(fwd_ctl()->after_resume);
  }
}

// A function of the application's CUDA driver, if it has loaded one: this
// never loads the driver.
static void *
driver_sym(const char *symbol)
{
  void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  return cuda != NULL ? dlsym(cuda, symbol) : NULL;
}

typedef int (*attr_fn)(void *, int, unsigned long long);

static attr_fn
pointer_get_attribute()
{
  static attr_fn attr = NULL;
  static bool looked = false;
  if (!looked) {
    attr = (attr_fn)driver_sym("cuPointerGetAttribute");
    looked = true;
  }
  return attr;
}

void
mana_cuda_use_context_of(const void *buf)
{
  typedef int (*set_fn)(void *);
  typedef int (*retain_fn)(void **, int);
  static set_fn set = NULL;
  static retain_fn retain = NULL;
  if (!mana_cuda_enabled() || buf == NULL) {
    return;
  }
  attr_fn attr = pointer_get_attribute();
  if (set == NULL) {
    set = (set_fn)driver_sym("cuCtxSetCurrent");
    retain = (retain_fn)driver_sym("cuDevicePrimaryCtxRetain");
  }
  if (attr == NULL || set == NULL || retain == NULL) {
    return;
  }
  void *ctx = NULL;
  int dev = -1;
  if (attr(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, (uintptr_t)buf) == 0 &&
      ctx != NULL) {
    set(ctx);
  } else if (attr(&dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                  (uintptr_t)buf) == 0 &&
             retain(&ctx, dev) == 0) {
    // VMM memory belongs to no context: use its device's primary context,
    // which the CUDA runtime uses.
    set(ctx);
  }
}

bool
mana_cuda_is_device_memory(const void *buf)
{
  if (!mana_cuda_enabled() || buf == NULL) {
    return false;
  }
  attr_fn attr = pointer_get_attribute();
  unsigned type = 0;
  return attr != NULL &&
         attr(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (uintptr_t)buf) == 0 &&
         type == CU_MEMORYTYPE_DEVICE;
}
