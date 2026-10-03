#include <stdio.h>
#include <stdlib.h>
#include "ucx_cuda_stubs.h"
#include "switch-context.h"
#include "lower-half-api.h"


static UpperCudaTable* get_table(const char* name) {
	if (lh_info->uh_cuda_table == NULL || lh_info->uh_fs == 0) {
		fprintf(stderr, "Cuda stub %s called before setup\n", name);
		return NULL;
	}
	return ((UpperCudaTable*) lh_info->uh_cuda_table);
}

extern "C" CUresult cuPointerGetAttribute(void* data, CUpointer_attribute attr, CUdeviceptr ptr) {
	UpperCudaTable* t = get_table("cuPointerGetAttribute");
	if (!t) return CUDA_ERROR_NOT_INITIALIZED;
	CUresult r;
	JUMP_TO_UPPER_HALF(lh_info->uh_fs);
	r = t->cuPointerGetAttribute(data, attr, ptr);
	RETURN_TO_LOWER_HALF();
	return r;
}
