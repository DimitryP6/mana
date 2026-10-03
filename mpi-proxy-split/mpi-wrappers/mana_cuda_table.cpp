#include <dlfcn.h>
#include <stdio.h>
#include "mana_cuda_table.h"
#include "lower-half-api.h"


static UpperCudaTable cuda_table;


// Helper macro to be used to load the matching CUDA functino into the table.
#define LOAD_CUDA_FN(fn) \
	cuda_table.fn = (decltype(cuda_table.fn))dlsym(lib, #fn); \
	if (cuda_table.fn == NULL) { \
		fprintf(stderr, "Cannot find %s in libcuda: %s\n", #fn, dlerror()); \
		return -1; \
	} 

int upper_half_init_cuda_table() {
	void* lib = dlopen("libcuda.so.1", RTLD_NOW);
	if (lib == NULL) {
		fprintf(stderr, "Cannot open libcuda.so.1: %s\n", dlerror());
		return -1;
	}
	// Load the functions into the table.
	LOAD_CUDA_FN(cuPointerGetAttribute);
	// Let the lower half know where the table is located.
	lh_info->uh_cuda_table = &cuda_table;
	return 0;
}
