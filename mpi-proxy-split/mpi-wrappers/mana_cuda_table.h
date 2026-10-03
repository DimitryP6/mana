#ifndef MANA_CUDA_TABLE_H
#define MANA_CUDA_TABLE_H



#include "lower-half/cuda-wrappers/ucx_cuda_stubs.h"


/**
 * Setup the cuda table with function pointers to the upper half's cuda functions.
 * @return -1 on error, 0 on success.
 */
int upper_half_init_cuda_table();


#endif // MANA_CUDA_TABLE_H
