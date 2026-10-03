#ifndef UCX_CUDA_STUBS_H
#define UCX_CUDA_STUBS_H


#include <cuda.h>


/**
 * An UpperCudaTable is a structure of CUDA Driver library functions,
 * where each functions points to the upper half in MANA's split 
 * process architecture.
 */
typedef struct {
	CUresult (*cuPointerGetAttribute)(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
} UpperCudaTable;


#endif // UCX_CUDA_STUBS_H
