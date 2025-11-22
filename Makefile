###
### For COMS E6998 Spring 2024
### Instructor: Kaoutar El Maghraoui
### Makefile for CUDA1 assignment
### By Wim Bohm, Waruna Ranasinghe, and Louis Rabiet
### Created: 2011-01-27 DVN
### Last Modified: Nov 2014 WB, WR, LR
###

SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
CXX := g++
CXXFLAGS := -O3

LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
INCLUDES := -IPart-A
OPTIONS   :=  -O3 -Xcompiler -fPIC  
#--maxrregcount=100 --ptxas-options -v 
LDFLAGS   := -Xlinker -no-pie

TAR_FILE_NAME  := BenCynaCUDA1.tar
EXECS :=  vecadd00 matmult00 matmult01 vecadd01 qB1 qB2 qB3 c1 c2 c3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) \
		Makefile report.pdf \
		Part-A/*.cu Part-A/*.h \
		Part-B/*.cpp Part-B/*.cu Part-B/*.py \
		Part-C/*.cu
#######################################################################

timer.o: Part-A/timer.cu Part-A/timer.h
	$(NVCC) -c $< -o $@ $(OPTIONS)

#######################################################################
vecaddKernel00.o: Part-A/vecaddKernel00.cu
	$(NVCC) -c $< -o $@ $(OPTIONS)

vecadd00: Part-A/vecadd.cu Part-A/vecaddKernel.h vecaddKernel00.o timer.o
	$(NVCC) Part-A/vecadd.cu vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
vecaddKernel01.o: Part-A/vecaddKernel01.cu
	$(NVCC) -c $< -o $@ $(OPTIONS)

vecadd01: Part-A/vecadd.cu Part-A/vecaddKernel.h vecaddKernel01.o timer.o
	$(NVCC) Part-A/vecadd.cu vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
## Provided Kernel
matmultKernel00.o: Part-A/matmultKernel00.cu Part-A/matmultKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS)

matmult00: Part-A/matmult.cu Part-A/matmultKernel.h matmultKernel00.o timer.o
	$(NVCC) Part-A/matmult.cu matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
# -------------------- Part-B Q1: CPU add two arrays --------------------
# q1.cpp -> q1 (no CUDA libs; pure C++)
qB1: Part-B/q1.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

qB2: Part-B/q2.cu
	$(NVCC) $< -o $@ $(OPTIONS)

qB3: Part-B/q3.cu
	$(NVCC) $< -o $@ $(OPTIONS)


#######################################################################
## Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
matmultKernel01.o: Part-A/matmultKernel01.cu Part-A/matmultKernel.h
	$(NVCC) -c $< -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult01: Part-A/matmult.cu Part-A/matmultKernel.h matmultKernel01.o timer.o
	$(NVCC) Part-A/matmult.cu matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32	

#######################################################################
# -------------------- Part-C --------------------
c1: Part-C/c1.cu
	$(NVCC) $< -o $@ $(OPTIONS)

c2: Part-C/c2.cu
	$(NVCC) $< -o $@ $(OPTIONS)

c3: Part-C/c3.cu
	$(NVCC) $< -o $@ $(OPTIONS) -lcudnn

# vecadd.cu intitializes the gpus ready for vector addition
# vecaddKernal00 hosts the code each thread will execute 
# the timer files are used to assess how fast the program runs