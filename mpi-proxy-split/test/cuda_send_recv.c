#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define N 1024

int main(int argc, char** argv) {
	// Set MPI rank.
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// Set CUDA device number.
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	cudaSetDevice(rank % dev_count);
	// Allocate host-side and device-side memory.
	int* host_buf = malloc(N * sizeof(int));
	int* dev_buf;
	cudaMalloc(((void**) &dev_buf), N * sizeof(int));
	// If main rank, set up host, copy to devices, and send.
	if (rank == 0) {
		for (int i = 0; i < N; i++) {
			host_buf[i] = i + 1;
		}
		cudaMemcpy(dev_buf, host_buf, N * sizeof(int), cudaMemcpyHostToDevice);
		MPI_Send(dev_buf, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
		printf("Rank 0 just sent data from device ptr %p\n", ((void*) dev_buf));
	} else {
		MPI_Recv(dev_buf, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		cudaMemcpy(host_buf, dev_buf, N * sizeof(int), cudaMemcpyDeviceToHost);
		// Check result.
		int mismatch = 0;
		for (int i = 0; i < N; i++) {
			if (host_buf[i] != (i + 1)) {
				fprintf(stderr, "Mismatch at %d: got %d but want %d\n",
						i, host_buf[i], i + 1);
				
				mismatch = 1;
			}
		}
		printf("Rank 1 receive %s\n", mismatch ? "FAILED" : "PASSED");
	}
	// Free memory.
	cudaFree(dev_buf);
	free(host_buf);
	MPI_Finalize();
	return 0;
}
