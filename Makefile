CC:=mpic++
NVCC:=nvcc
CFLAGS:=-lm -O3
NVCFLAGS:= -gencode=arch=compute_75,code=compute_75 
OBJ:=eval.o

%.o: %.cu
	$(NVCC) $< $(NVCFLAGS) -o $@ 

all: NVCFLAGS +=-O3
all: $(OBJ)

debug: NVCFLAGS +=-g -G -DDEBUG
debug: $(OBJ)

clean:
	rm -rf *.o
