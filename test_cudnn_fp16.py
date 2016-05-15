
import math
import numpy as np
import pycuda.autoinit
import libcudnn
from gputensor import GPUTensor

xh = np.array( [[[[ 1 + m * x for x in list(range(4)) ] for m in range(4) ]]], dtype=np.float16 )
yh = xh + 0.5
print(xh)
print(yh)
print(xh.shape)

cudnn_context = libcudnn.cudnnCreate()

x = GPUTensor(xh)
y = GPUTensor(yh)
print(x.dtype, y.dtype)

x_desc = x.get_cudnn_tensor_desc()
y_desc = y.get_cudnn_tensor_desc()
print(x_desc)
libcudnn.cudnnAddTensor(cudnn_context, 1.0, x_desc.ptr, x.get_gpu_voidp(),
        1.0,  y_desc.ptr, y.get_gpu_voidp())

yh2 = y.get()
print(y)
print(y.dtype)
