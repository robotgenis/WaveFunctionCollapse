import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Define the size of the input data
n = 1024

# Define the CUDA kernel that will use gridDim
mod = SourceModule("""
  __global__ void print_block_index(int n)
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n)
    {
      int block_index = gridDim.x * blockIdx.y + blockIdx.x;
      printf("Block index: %d\\n", block_index);
    }
  }
""")

# Get a handle to the CUDA kernel
print_block_index = mod.get_function("print_block_index")

# Define the size of the blocks and grids to use for the kernel
block_size = (256, 1, 1)
grid_size = (n // block_size[0], 1, 1)

n_gpu = cuda.mem_alloc(np.int32(n).nbytes)
cuda.memcpy_htod(n_gpu, np.int32(n))

# Call the CUDA kernel to print the block index for each block
print_block_index(n_gpu, block=block_size, grid=grid_size)
