import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Set up the data
a = np.float32(2.0)
b = np.float32(3.0)

# Transfer the data to the GPU
a_gpu = drv.mem_alloc(a.nbytes)
drv.memcpy_htod(a_gpu, a)
b_gpu = drv.mem_alloc(b.nbytes)
drv.memcpy_htod(b_gpu, b)

# Create a CUDA kernel
mod = SourceModule("""
    __global__ void add(float *a, float *b, float *c)
    {
      *c = *a * *b;
    }
""")
add = mod.get_function("add")

# Allocate space for the result on the GPU and run the kernel
c_gpu = drv.mem_alloc(a.nbytes)
add(a_gpu, b_gpu, c_gpu, block=(1, 1, 1), grid=(1, 1))

# Transfer the result back to the host
c = np.empty_like(a)
drv.memcpy_dtoh(c, c_gpu)

# Print the result
print(c)
