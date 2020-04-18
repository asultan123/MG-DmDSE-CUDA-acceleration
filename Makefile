CC:=mpic++
NVCC:=nvcc
CFLAGS:=-lm -O3
NVCFLAGS:= -g -gencode=arch=compute_75,code=compute_75 
OBJ:=part2.o

%.o: %.cpp
	$(CC) $< $(CFLAGS) -o $@ 

%.o: %.cu
	$(NVCC) $< $(NVCFLAGS) -o $@ 

all: $(OBJ)

single: all
	./part1.o

sim_part1: all
	mpirun -n $(CORES) part1.o

sim_part2a: all
	mpirun -n $(CORES) part2a.o

sim_part2b: all
	mpirun -n $(CORES) part2b.o

clean:
	rm -rf *.o
