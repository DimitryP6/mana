# CUDA_HOME: the toolkit (module load CUDA sets it).  CUPTI is in the toolkit's
# include/lib64, or in extras/CUPTI on some installs; both are searched.
CUDA_HOME ?= /usr/local/cuda
MPICC ?= mpicc
CUPTI_INC = -I$(CUDA_HOME)/include -I$(CUDA_HOME)/extras/CUPTI/include
CUPTI_LIB = -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/extras/CUPTI/lib64 \
            -Wl,-rpath,$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64
CFLAGS = -O2 -g -fPIC -Wall -Wextra

all: libcudatrace.so libmpitrace.so

# Not linked with -lcuda: the driver loads this library itself.
libcudatrace.so: cudatrace.c
	$(CC) $(CFLAGS) -shared -o $@ $< $(CUPTI_INC) $(CUPTI_LIB) -lcupti -lpthread

libmpitrace.so: mpitrace.c
	$(MPICC) $(CFLAGS) -shared -o $@ $<

clean:
	rm -f libcudatrace.so libmpitrace.so

.PHONY: all clean
