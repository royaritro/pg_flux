EXTENSION = pg_gftr
MODULE_big = pg_gftr
DATA = sql/pg_gftr--0.1.0.sql
PGFILEDESC = "pg_gftr - experimental CUDA joins and aggregation"

PG_CONFIG ?= pg_config
USE_CUDA ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= 80
NVCCFLAGS ?= -O3 -std=c++17

OBJS = src/pg_gftr.o src/pg_gftr_join.o src/pg_gftr_parallel.o src/gftr_cpu.o src/gftr_join_cpu.o
PG_CPPFLAGS = -Isrc
ifeq ($(USE_CUDA),1)
OBJS += src/gftr_cuda.o src/gftr_join_cuda.o
PG_CPPFLAGS += -DGFTR_WITH_CUDA
SHLIB_LINK += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
else
OBJS += src/gftr_cuda_stub.o src/gftr_join_cuda_stub.o
endif

# Do not ask LLVM's C bitcode rule to compile CUDA translation units.
override with_llvm = no
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

src/gftr_cuda.o: src/gftr_cuda.cu src/gftr_backend.h
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-fPIC -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH) -Isrc -c $< -o $@

src/gftr_join_cuda.o: src/gftr_join_cuda.cu src/gftr_join.h src/gftr_backend.h src/gftr_join_lookup.h
	$(NVCC) $(NVCCFLAGS) -Xcompiler=-fPIC -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH) -Isrc -c $< -o $@

src/pg_gftr.o src/pg_gftr_join.o src/gftr_join_cpu.o src/gftr_join_cuda_stub.o: src/gftr_join.h
src/pg_gftr.o src/pg_gftr_join.o src/pg_gftr_parallel.o: src/pg_gftr_fused.h
src/gftr_join_cpu.o: src/gftr_join_lookup.h
src/pg_gftr.o src/pg_gftr_join.o src/gftr_cpu.o src/gftr_cuda_stub.o src/gftr_join_cpu.o src/gftr_join_cuda_stub.o: src/gftr_backend.h

.PHONY: test-cpu test-cuda
test-cpu:
	$(CC) -std=c11 -D_POSIX_C_SOURCE=200809L -O2 -Wall -Wextra -Isrc tests/backend_test.c src/gftr_cpu.c -o tests/backend_test
	./tests/backend_test
	$(CC) -std=c11 -D_POSIX_C_SOURCE=200809L -O2 -Wall -Wextra -Isrc tests/join_backend_test.c src/gftr_join_cpu.c src/gftr_join_cuda_stub.c -o tests/join_backend_test
	./tests/join_backend_test

test-cuda: src/gftr_join_cuda.o
	$(CC) -std=c11 -D_POSIX_C_SOURCE=200809L -O2 -Isrc -c src/gftr_cpu.c -o tests/gftr_cpu_test.o
	$(NVCC) $(NVCCFLAGS) -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -Isrc tests/cuda_test.cu src/gftr_cuda.cu tests/gftr_cpu_test.o -o tests/cuda_test
	./tests/cuda_test
	$(CC) -std=c11 -D_POSIX_C_SOURCE=200809L -O2 -Isrc -c tests/join_backend_test.c -o tests/join_backend_test.o
	$(CC) -std=c11 -D_POSIX_C_SOURCE=200809L -O2 -Isrc -c src/gftr_join_cpu.c -o tests/gftr_join_cpu_test.o
	$(NVCC) $(NVCCFLAGS) src/gftr_join_cuda.o tests/join_backend_test.o tests/gftr_join_cpu_test.o -o tests/join_backend_test_cuda
	./tests/join_backend_test_cuda --cuda

EXTRA_CLEAN += tests/backend_test tests/cuda_test tests/gftr_cpu_test.o src/gftr_cuda.o src/gftr_cuda_stub.o
EXTRA_CLEAN += tests/join_backend_test tests/join_backend_test_cuda tests/join_backend_test.o tests/gftr_join_cpu_test.o src/gftr_join_cuda.o src/gftr_join_cuda_stub.o
