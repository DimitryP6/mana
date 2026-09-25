#include <cuda_runtime.h>


#ifndef MANA_CUDA
#define MANA_CUDA


/**
 * Given a buffer pointer, determines if it is a CUDA device pointer.
 * @param p is a pointer to an unmodifiable type.
 * @return whether the pointer is a device or non-device pointer.
 */
bool mana_cuda_is_dev_ptr(const void* p);


#endif // MANA_CUDA
