from numba import cuda
import numpy
from numpy import random

@cuda.jit
def my_kernel(io_array, o_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    
    # Compute flattened index inside the array
    pos = tx + ty * bw

    num = 0
    count = 0  
   
    if pos < io_array.size:
      num = io_array[pos]
      for nu in io_array:
        if num == nu:
          count += 1
      o_array[num] = count
      count = 0
     

   

data = random.randint(10, size=(1024))

o_array = numpy.zeros([10])

threads_per_block = 32

blocks_per_grid = ( data.size + (threads_per_block - 1) )

# iniciando o kernel
my_kernel[blocks_per_grid, threads_per_block](data, o_array )


print("vetor")
print(data)
print("vetor soma")
print(o_array)
