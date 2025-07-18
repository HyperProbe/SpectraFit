CUDA_PATH = /usr/local/cuda-11.3

CC = $(CUDA_PATH)/bin/nvcc
PROGRAM = CUDAMCML
CFLAGS = -arch=sm_60 -O3 -Xptxas -v
IFLAGS = -I $(CUDA_PATH)/include -I $(CUDA_PATH)/samples/common/inc
LFLAGS = -L $(CUDA_PATH)/lib -L $(CUDA_PATH)/samples/common/lib/linux -L $(HOME)/lib64 -lcuda -lcudart # -lcutil
SOURCE = ./src/CUDAMCMLmain.cu
DEP = ./src/CUDAMCMLmain.cu ./src/CUDAMCMLio.cu ./src/CUDAMCMLmem.cu ./src/CUDAMCMLrng.cu ./src/CUDAMCMLtransport.cu ./src/CUDAMCML.h

$(PROGRAM): $(DEP)
	$(CC) $(CFLAGS) $(SOURCE) $(IFLAGS) $(LFLAGS) -o $(PROGRAM)

clean:
	\rm -f *~ *.o *.linkinfo CUDAMCML



