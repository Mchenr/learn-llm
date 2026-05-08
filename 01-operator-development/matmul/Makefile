NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_89 -Iinclude

TARGETS = 01_naive_cuda 02_tiled_cuda 03_regtiling_cuda 04_wmma_cuda 05_cublas

all: $(TARGETS)

01_naive_cuda: 01_naive_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

02_tiled_cuda: 02_tiled_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

03_regtiling_cuda: 03_regtiling_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

04_wmma_cuda: 04_wmma_cuda.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

05_cublas: 05_cublas.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< -lcublas

clean:
	rm -f $(TARGETS)

.PHONY: all clean
