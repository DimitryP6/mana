#include "mana_cuda.h"


bool mana_cuda_is_dev_ptr(const void* p) {
	if (p == nullptr) {
		return false;
	}
	cudaPointerAttributes attr;
	cudaError_t err = cudaPointerGetAttributes(&attr, p);
	if (err != cudaSuccess) {
		// Clear error from older versions if host.
		cudaGetLastError();
		return false;
	}
	return (attr.type == cudaMemoryTypeDevice);
}
