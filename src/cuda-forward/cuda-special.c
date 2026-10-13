/*
 * The hand-written functions of the libcuda.so.1 shim; the others are
 * generated trampolines (gen-forward-shim.py).  See README.md.
 *
 *  - cuGetProcAddress and cuGetExportTable hand out upper-half function
 *    pointers, which the lower half must call through trampolines.
 *  - Host-memory registrations are released before a checkpoint.
 *  - Buffer ids change on each resume.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>

#include "fwd-runtime.h"

// 'symbol' of the upper half's libcuda.so.1.
static void *
upper_sym(const char *symbol)
{
  void *p = NULL;
  FWD_CALL_UPPER(p, NULL, mana_fwd_ctl.uh_dlsym("libcuda.so.1", symbol));
  return p;
}

// static __typeof__(&name) var = the upper half's 'name'.
#define UPPER(var, name) \
  static __typeof__(&name) var = NULL; \
  if (var == NULL) { var = (__typeof__(&name))upper_sym(#name); }
// rc = call, an upper-half function.
#define CALL(rc, call) FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE, call)

static void
vlog(const char *fmt, ...)
{
  if (mana_fwd_verbose()) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[mana-fwd %d] ", getpid());
    vfprintf(stderr, fmt, ap);
    va_end(ap);
  }
}

/*
 * Function pointers.  cuGetProcAddress (and cudaGetDriverEntryPoint of the
 * libcudart shim) return upper-half functions: the lower half gets a
 * trampoline instead, or this shim's own function for those that it
 * defines itself.
 */
static void *own_function(const char *symbol, void *upper_fn);

static void *
wrap_driver_fn(const char *symbol, void *upper_fn)
{
  void *own = own_function(symbol, upper_fn);
  return own != NULL ? own : mana_fwd_wrap_ptr(upper_fn, symbol);
}

CUresult
cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                    cuuint64_t flags,
                    CUdriverProcAddressQueryResult *symbolStatus)
{
  UPPER(real, cuGetProcAddress_v2);
  if (real == NULL) {
    return CUDA_ERROR_NO_DEVICE;
  }
  CUresult rc;
  CALL(rc, real(symbol, pfn, cudaVersion, flags, symbolStatus));
  if (rc == CUDA_SUCCESS && pfn != NULL && *pfn != NULL) {
    *pfn = wrap_driver_fn(symbol, *pfn);
  }
  return rc;
}

// The API before CUDA 12.0, without symbolStatus.
#undef cuGetProcAddress
CUresult
cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                 cuuint64_t flags)
{
  typedef CUresult (*v1_fn)(const char *, void **, int, cuuint64_t);
  static v1_fn real = NULL;
  if (real == NULL) {
    real = (v1_fn)upper_sym("cuGetProcAddress");
    if (real == NULL) {
      return CUDA_ERROR_NO_DEVICE;
    }
  }
  CUresult rc;
  CALL(rc, real(symbol, pfn, cudaVersion, flags));
  if (rc == CUDA_SUCCESS && pfn != NULL && *pfn != NULL) {
    *pfn = wrap_driver_fn(symbol, *pfn);
  }
  return rc;
}

/*
 * The driver's private function tables, which the CUDA runtime (e.g. the
 * static one in libnccl) uses besides the public functions.
 */
CUresult
cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId)
{
  UPPER(real, cuGetExportTable);
  if (real == NULL) {
    return CUDA_ERROR_NO_DEVICE;
  }
  const void *t = NULL;
  CUresult rc;
  CALL(rc, real(&t, pExportTableId));
  if (rc == CUDA_SUCCESS && t != NULL) {
    t = mana_fwd_wrap_table(t, (void *)real, pExportTableId);
  }
  *ppExportTable = t;
  return rc;
}

/*
 * Host-memory registrations of the lower half (cuMemHostRegister and
 * cudaHostRegister, by UCX or the MPI).  The CUDA checkpoint saves them, and
 * the restore registers them again, at their addresses: at restart that is
 * a new lower half, and the restore fails.  So they are released before a
 * checkpoint and registered again on resume, and the lower half sees no
 * change.  At restart, the new lower half has registered nothing yet.
 */
static CUcontext
current_ctx(void)
{
  UPPER(get, cuCtxGetCurrent);
  CUcontext ctx = NULL;
  CUresult rc = CUDA_ERROR_NO_DEVICE;
  if (get != NULL) {
    CALL(rc, get(&ctx));
  }
  return rc == CUDA_SUCCESS ? ctx : NULL;
}

// Registers (reg = 1) or unregisters 'h' in its context.
static CUresult
host_reg_call(HostReg *h, int reg)
{
  UPPER(push, cuCtxPushCurrent_v2);
  UPPER(pop, cuCtxPopCurrent_v2);
  UPPER(hreg, cuMemHostRegister_v2);
  UPPER(hunreg, cuMemHostUnregister);
  CUresult rc, ignored;
  CUcontext old;
  CALL(ignored, push(h->ctx));
  if (reg) {
    CALL(rc, hreg(h->ptr, h->size, h->flags));
  } else {
    CALL(rc, hunreg(h->ptr));
  }
  CALL(ignored, pop(&old));
  (void)ignored;
  return rc;
}

#define HOST_REGISTER_WRAPPER(name)                                           \
CUresult                                                                      \
name(void *p, size_t bytesize, unsigned int Flags)                            \
{                                                                             \
  UPPER(real, name);                                                          \
  if (real == NULL) {                                                         \
    return CUDA_ERROR_NO_DEVICE;                                              \
  }                                                                           \
  CUresult rc;                                                                \
  CALL(rc, real(p, bytesize, Flags));                                         \
  if (rc == CUDA_SUCCESS) {                                                   \
    mana_fwd_note_host_register(p, bytesize, Flags, current_ctx());           \
  }                                                                           \
  return rc;                                                                  \
}

#undef cuMemHostRegister
HOST_REGISTER_WRAPPER(cuMemHostRegister)
HOST_REGISTER_WRAPPER(cuMemHostRegister_v2)

CUresult
cuMemHostUnregister(void *p)
{
  UPPER(real, cuMemHostUnregister);
  if (real == NULL) {
    return CUDA_ERROR_NO_DEVICE;
  }
  CUresult rc;
  CALL(rc, real(p));
  mana_fwd_note_host_unregister(p);
  return rc;
}

/*
 * Pointer attributes.
 *
 * CU_POINTER_ATTRIBUTE_BUFFER_ID: on resume, the GPU's checkpoint and
 * restore may have moved device memory to other physical pages, but the
 * driver keeps each allocation's buffer id.  The MPI's caches check their
 * entries by buffer id: libfabric's CUDA monitor keeps NIC registrations, so
 * after a resume the NIC would still access the old pages.  So the buffer ids
 * that the lower half sees change on each resume.  (At restart, the lower
 * half and its caches are new.)
 */
static unsigned long long ckpt_generation;  // bumped on each resume

static void
adjust_attribute(CUpointer_attribute attribute, void *data, CUdeviceptr p)
{
  if (data != NULL && attribute == CU_POINTER_ATTRIBUTE_BUFFER_ID) {
    *(unsigned long long *)data ^= ckpt_generation << 48;
  }
}

CUresult
cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
                      CUdeviceptr ptr)
{
  UPPER(real, cuPointerGetAttribute);
  CUresult rc;
  CALL(rc, real(data, attribute, ptr));
  if (rc == CUDA_SUCCESS) {
    adjust_attribute(attribute, data, ptr);
  }
  return rc;
}

CUresult
cuPointerGetAttributes(unsigned int numAttributes,
                       CUpointer_attribute *attributes, void **data,
                       CUdeviceptr ptr)
{
  UPPER(real, cuPointerGetAttributes);
  CUresult rc;
  CALL(rc, real(numAttributes, attributes, data, ptr));
  for (unsigned i = 0; rc == CUDA_SUCCESS && i < numAttributes; i++) {
    adjust_attribute(attributes[i], data[i], ptr);
  }
  return rc;
}

/*
 * The checkpoint hooks (FwdCtl).
 */
static void
cuda_before_ckpt(void)
{
  int n;
  HostReg *regs = mana_fwd_host_regs(&n);
  int released = 0;
  for (int i = 0; i < n; i++) {
    if (regs[i].on) {
      CUresult rc = host_reg_call(&regs[i], 0);
      if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[mana-fwd] cuMemHostUnregister(%p): %d\n",
                regs[i].ptr, rc);
      }
      released++;
    }
  }
  vlog("before checkpoint: %d host registrations released\n", released);
}

static void
cuda_after_resume(void)
{
  ckpt_generation++;
  int n;
  HostReg *regs = mana_fwd_host_regs(&n);
  for (int i = 0; i < n; i++) {
    if (regs[i].on) {
      CUresult rc = host_reg_call(&regs[i], 1);
      if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[mana-fwd] cuMemHostRegister(%p): %d\n",
                regs[i].ptr, rc);
      }
    }
  }
}

__attribute__((constructor)) static void
cuda_special_init(void)
{
  mana_fwd_cuda_before_ckpt = cuda_before_ckpt;
  mana_fwd_cuda_after_resume = cuda_after_resume;
  mana_fwd_wrap_driver_fn = wrap_driver_fn;
}

// This shim's own function for 'symbol', which the driver resolved to
// 'upper_fn', or NULL.
static void *
own_function(const char *symbol, void *upper_fn)
{
  static const struct { const char *name; void *fn; } own[] = {
    { "cuGetExportTable", (void *)cuGetExportTable },
    { "cuMemHostRegister", (void *)cuMemHostRegister_v2 },
    { "cuMemHostUnregister", (void *)cuMemHostUnregister },
    { "cuPointerGetAttribute", (void *)cuPointerGetAttribute },
    { "cuPointerGetAttributes", (void *)cuPointerGetAttributes },
  };
  // The two versions of cuGetProcAddress differ in their arguments: the
  // driver's answer tells which one the caller asked for.
  if (strcmp(symbol, "cuGetProcAddress") == 0) {
    return upper_fn == upper_sym("cuGetProcAddress_v2")
             ? (void *)cuGetProcAddress_v2 : (void *)cuGetProcAddress;
  }
  for (unsigned i = 0; i < sizeof own / sizeof own[0]; i++) {
    if (strcmp(symbol, own[i].name) == 0) {
      return own[i].fn;
    }
  }
  return NULL;
}
