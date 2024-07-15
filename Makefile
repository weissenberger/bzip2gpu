NVCC = nvcc

LDFLAGS = -lstdc++fs -lgomp

NVCCFLAGS = -std=c++17 -O3 -arch=native -Xcompiler "-Wall -std=c++17 -O3 -march=native -fopenmp"

EXEC_NAME = bz2gpu

INCLUDE_DIR = include
SRC_DIR = src
BIN_DIR = bin

SRC_FILES := $(wildcard $(SRC_DIR)/*.cc)
CU_SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cc,$(BIN_DIR)/%.o,$(SRC_FILES))
CU_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%.obj,$(CU_SRC_FILES))

default: link_cuda

link_cuda: cuda cpp
	$(NVCC) $(OBJ_FILES) $(CU_OBJ_FILES) $(LDFLAGS) -o $(BIN_DIR)/$(EXEC_NAME)

cpp: $(OBJ_FILES)
cuda: $(CU_OBJ_FILES)

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cc
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDE_DIR) -c -o $@ $<

$(BIN_DIR)/%.obj: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -I $(INCLUDE_DIR) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(BIN_DIR)/*.o
	rm -f $(BIN_DIR)/*.obj
	rm -f $(BIN_DIR)/$(EXEC_NAME)
