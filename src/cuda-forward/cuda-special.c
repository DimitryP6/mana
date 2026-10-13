/*
 * The hand-written functions of the libcuda.so.1 shim; the others are
 * generated trampolines (gen-forward-shim.py).  See README.md.
 *
 *  - cuGetProcAddress and cuGetExportTable hand out upper-half function
 *    pointers, which the lower half must call through trampolines.
 *  - Host-memory registrations are released before a checkpoint.
 *  - Virtual CUDA IPC handles let the MPI share device memory between
 *    processes, and still checkpoint.
 *  - Buffer ids change on each resume.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
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
 * Virtual CUDA IPC handles.
 *
 * A legacy CUDA IPC export (cuIpcGetMemHandle, of cudaMalloc memory) makes
 * the process impossible to checkpoint until the memory is freed, and no IPC
 * handle or mapping survives a checkpoint.  VMM memory exported as a POSIX
 * fd can be checkpointed, if no importer has it mapped.  So the lower half
 * (UCX's cuda_ipc, Cray's GTL) gets virtual legacy IPC handles of VMM memory
 * (made by libmana_cuda_vmm.so in the upper half):
 *  - VMM memory exportable as a POSIX fd is reported as legacy-IPC capable
 *    (cuPointerGetAttribute(s)), and cudaMalloc memory is not: the MPI then
 *    shares the one with the legacy API, and copies the other;
 *  - cuIpcGetMemHandle exports the allocation as a POSIX fd (once) and
 *    returns a handle that names it: exporter pid, fd number, buffer id;
 *  - cuIpcOpenMemHandle gets the fd with pidfd_getfd(), imports it, and maps
 *    it in a VA range reserved for it, which the importer keeps (UCX's IPC
 *    cache stays valid);
 *  - before a checkpoint, importers unmap (the VA stays reserved), and
 *    exporters close their exports, but keep the fd numbers;
 *  - on resume, exporters export again on the same fd numbers; then, after
 *    a global barrier, importers import again and map at the same VA;
 *  - at restart, the lower half and its IPC cache are new: nothing to do.
 */
#define VIPC_MAGIC 0x3243504941414e4dULL      // "MANAIPC2"
#define MAX_EXPORTS 1024
#define MAX_IMPORTS 1024

typedef struct {
  unsigned long long magic;
  int pid;                    // the exporter (real pid)
  int fd;                     // its export, stable across checkpoints
  unsigned long long size;
  unsigned long long id;      // CU_POINTER_ATTRIBUTE_BUFFER_ID
} VirtIpcHandle;              // fits in CUipcMemHandle (64 bytes)

typedef struct {
  CUdeviceptr base;
  size_t size;
  unsigned long long id;
  int fd;
  int on;
} Export;
typedef struct {
  CUdeviceptr va;
  size_t size;
  int pid, fd;
  unsigned long long id;
  CUdevice dev;
  CUmemGenericAllocationHandle h;
  int mapped, on;
} Import;
static Export exports[MAX_EXPORTS];
static int n_exports;
static Import imports[MAX_IMPORTS];
static int n_imports;

// MANA_FWD_VIRTUAL_IPC=0: the real legacy IPC (no checkpoint then).
static int
vipc_enabled(void)
{
  static int on = -1;
  if (on < 0) {
    const char *v = getenv("MANA_FWD_VIRTUAL_IPC");
    on = v == NULL || v[0] != '0';
  }
  return on;
}

CUresult
cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr)
{
  UPPER(real, cuMemRetainAllocationHandle);
  CUresult rc;
  CALL(rc, real(handle, addr));
  return rc;
}

// The buffer id of the allocation that contains 'p', or 0.
static unsigned long long
buffer_id(CUdeviceptr p)
{
  UPPER(attr, cuPointerGetAttribute);
  unsigned long long id = 0;
  CUresult rc;
  CALL(rc, attr(&id, CU_POINTER_ATTRIBUTE_BUFFER_ID, p));
  return rc == CUDA_SUCCESS ? id : 0;
}

// A fresh POSIX-fd export of the VMM allocation at 'base', or -1.
static int
export_fd(CUdeviceptr base)
{
  UPPER(retain, cuMemRetainAllocationHandle);
  UPPER(exportfd, cuMemExportToShareableHandle);
  UPPER(release, cuMemRelease);
  CUmemGenericAllocationHandle h;
  CUresult rc;
  CALL(rc, retain(&h, (void *)base));
  if (rc != CUDA_SUCCESS) {
    return -1;                // not VMM memory
  }
  int fd = -1;
  CALL(rc, exportfd(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  CUresult ignored;
  CALL(ignored, release(h));
  (void)ignored;
  return rc == CUDA_SUCCESS ? fd : -1;
}

// Whether the memory of export 'e' is still the allocation it exported:
// the application may have freed it, and reused the address.
static int
export_alive(const Export *e)
{
  return buffer_id(e->base) == e->id;
}

static void
drop_export(Export *e)
{
  close(e->fd);
  e->on = 0;
}

CUresult
cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr)
{
  UPPER(real, cuIpcGetMemHandle);
  UPPER(range, cuMemGetAddressRange_v2);
  CUresult rc;
  if (!vipc_enabled()) {
    CALL(rc, real(pHandle, dptr));
    return rc;
  }
  CUdeviceptr base;
  size_t size;
  CALL(rc, range(&base, &size, dptr));
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  Export *e = NULL;
  for (int i = 0; i < n_exports && e == NULL; i++) {
    if (exports[i].on && exports[i].base == base) {
      if (export_alive(&exports[i])) {
        e = &exports[i];
      } else {
        drop_export(&exports[i]);
      }
    }
  }
  if (e == NULL) {
    int i;
    for (i = 0; i < n_exports && exports[i].on; i++) {
    }
    int fd = i < MAX_EXPORTS ? export_fd(base) : -1;
    if (fd < 0) {
      // cudaMalloc memory: a real legacy export would make the process
      // impossible to checkpoint.  The MPI falls back to a copy.
      vlog("no virtual IPC for %#llx (not VMM POSIX-fd memory)\n",
           (unsigned long long)base);
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    e = &exports[i];
    *e = (Export){ base, size, buffer_id(base), fd, 1 };
    if (i == n_exports) {
      n_exports++;
    }
    vlog("export %#llx (%zu bytes) as fd %d\n", (unsigned long long)base, size,
         fd);
  }
  VirtIpcHandle v = { VIPC_MAGIC, getpid(), e->fd, e->size, e->id };
  memset(pHandle, 0, sizeof *pHandle);
  memcpy(pHandle, &v, sizeof v);
  return CUDA_SUCCESS;
}

// Imports 'im' and maps it at its VA.
static CUresult
map_import(Import *im)
{
  UPPER(importfd, cuMemImportFromShareableHandle);
  UPPER(map, cuMemMap);
  UPPER(access, cuMemSetAccess);
  int pidfd = syscall(SYS_pidfd_open, im->pid, 0);
  int fd = pidfd >= 0 ? syscall(SYS_pidfd_getfd, pidfd, im->fd, 0) : -1;
  if (pidfd >= 0) {
    close(pidfd);
  }
  if (fd < 0) {
    vlog("pidfd_getfd(pid %d, fd %d): errno %d\n", im->pid, im->fd, errno);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  CUresult rc;
  CALL(rc, importfd(&im->h, (void *)(long)fd,
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  close(fd);
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  CALL(rc, map(im->va, im->size, 0, im->h, 0));
  if (rc == CUDA_SUCCESS) {
    CUmemAccessDesc acc;
    memset(&acc, 0, sizeof acc);
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id = im->dev;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CALL(rc, access(im->va, im->size, &acc, 1));
  }
  im->mapped = rc == CUDA_SUCCESS;
  return rc;
}

static void
unmap_import(Import *im)
{
  UPPER(unmap, cuMemUnmap);
  UPPER(release, cuMemRelease);
  CUresult ignored;
  if (im->mapped) {
    CALL(ignored, unmap(im->va, im->size));
    CALL(ignored, release(im->h));
    im->mapped = 0;
  }
  (void)ignored;
}

CUresult
cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                      unsigned int Flags)
{
  VirtIpcHandle v;
  memcpy(&v, &handle, sizeof v);
  CUresult rc;
  if (v.magic != VIPC_MAGIC) {
    UPPER(real, cuIpcOpenMemHandle_v2);
    CALL(rc, real(pdptr, handle, Flags));
    return rc;
  }
  UPPER(reserve, cuMemAddressReserve);
  UPPER(getdev, cuCtxGetDevice);
  int i;
  for (i = 0; i < n_imports; i++) {
    Import *im = &imports[i];
    if (im->on && im->pid == v.pid && im->fd == v.fd && im->id == v.id) {
      *pdptr = im->va;          // opened already
      return CUDA_SUCCESS;
    }
  }
  for (i = 0; i < n_imports && imports[i].on; i++) {
  }
  if (i == MAX_IMPORTS) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  Import *im = &imports[i];
  memset(im, 0, sizeof *im);
  im->pid = v.pid;
  im->fd = v.fd;
  im->id = v.id;
  im->size = v.size;
  CALL(rc, getdev(&im->dev));
  if (rc == CUDA_SUCCESS) {
    CALL(rc, reserve(&im->va, im->size, 0, 0, 0));
  }
  if (rc == CUDA_SUCCESS) {
    rc = map_import(im);
  }
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  im->on = 1;
  if (i == n_imports) {
    n_imports++;
  }
  vlog("import pid %d fd %d at %#llx\n", v.pid, v.fd,
       (unsigned long long)im->va);
  *pdptr = im->va;
  return CUDA_SUCCESS;
}

#undef cuIpcOpenMemHandle
CUresult
cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                   unsigned int Flags)
{
  return cuIpcOpenMemHandle_v2(pdptr, handle, Flags);
}

CUresult
cuIpcCloseMemHandle(CUdeviceptr dptr)
{
  for (int i = 0; i < n_imports; i++) {
    Import *im = &imports[i];
    if (im->on && im->va == dptr) {
      UPPER(freeva, cuMemAddressFree);
      unmap_import(im);
      CUresult ignored;
      CALL(ignored, freeva(im->va, im->size));
      (void)ignored;
      im->on = 0;
      return CUDA_SUCCESS;
    }
  }
  UPPER(real, cuIpcCloseMemHandle);
  CUresult rc;
  CALL(rc, real(dptr));
  return rc;
}

// Before a checkpoint: no import mapped, and no export open.  An export
// keeps its fd number (on /dev/null).
static void
vipc_before_ckpt(void)
{
  int ni = 0, ne = 0;
  for (int i = 0; i < n_imports; i++) {
    if (imports[i].on && imports[i].mapped) {
      unmap_import(&imports[i]);
      ni++;
    }
  }
  int null = open("/dev/null", O_RDONLY);
  for (int i = 0; i < n_exports; i++) {
    Export *e = &exports[i];
    if (e->on && !export_alive(e)) {
      drop_export(e);
    } else if (e->on) {
      dup2(null, e->fd);
      ne++;
    }
  }
  close(null);
  vlog("before checkpoint: %d imports unmapped, %d exports closed\n", ni, ne);
}

// On resume, step 1 (every process): export again, on the same fd numbers.
static void
vipc_after_resume_exports(void)
{
  for (int i = 0; i < n_exports; i++) {
    Export *e = &exports[i];
    if (e->on) {
      int fd = export_fd(e->base);
      if (fd < 0) {
        vlog("cannot export %#llx again\n", (unsigned long long)e->base);
        drop_export(e);
      } else {
        dup2(fd, e->fd);
        close(fd);
      }
    }
  }
}

// On resume, step 2 (after every process did step 1): import again, at the
// same VA.  An import whose exporter dropped the export stays unmapped
// until the MPI closes it.
static void
vipc_after_resume_imports(void)
{
  for (int i = 0; i < n_imports; i++) {
    Import *im = &imports[i];
    if (im->on && !im->mapped) {
      CUresult rc = map_import(im);
      if (rc != CUDA_SUCCESS) {
        vlog("cannot import pid %d fd %d again: %d\n", im->pid, im->fd,
             (int)rc);
      }
    }
  }
}

/*
 * Pointer attributes.
 *
 * CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE: see "Virtual CUDA IPC
 * handles".
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

static int
vmm_posix_fd_memory(CUdeviceptr p)
{
  UPPER(attr, cuPointerGetAttribute);
  unsigned long long types = 0;
  CUresult rc;
  CALL(rc, attr(&types, CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, p));
  return rc == CUDA_SUCCESS &&
         (types & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != 0;
}

static void
adjust_attribute(CUpointer_attribute attribute, void *data, CUdeviceptr p)
{
  if (data == NULL) {
    return;
  }
  if (attribute == CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE &&
      vipc_enabled()) {
    *(int *)data = vmm_posix_fd_memory(p);
  } else if (attribute == CU_POINTER_ATTRIBUTE_BUFFER_ID) {
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
  vipc_before_ckpt();
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
  vipc_after_resume_exports();
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
  mana_fwd_cuda_after_resume_peers = vipc_after_resume_imports;
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
    { "cuMemRetainAllocationHandle", (void *)cuMemRetainAllocationHandle },
    { "cuIpcGetMemHandle", (void *)cuIpcGetMemHandle },
    { "cuIpcOpenMemHandle", (void *)cuIpcOpenMemHandle_v2 },
    { "cuIpcCloseMemHandle", (void *)cuIpcCloseMemHandle },
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
