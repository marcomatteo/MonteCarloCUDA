objects = main.o Matrix.o Options.o MonteCarloHost.o MonteCarloKernel.o

all: $(objects)
	nvcc -arch=sm_35 $(objects) -o app

%.o: %.c
	nvcc -x cu -arch=sm_35 -I. -dc $< -o $@

clean:
	rm -f *.o app