
import math
import numpy as np
import pycuda.autoinit
import libcudnn
from gputensor import GPUTensor

dt = np.float32 

xh = np.ones((1,1,4,4), dtype=dt)
print(xh)
print(xh.shape)

cudnn_context = libcudnn.cudnnCreate()

x = GPUTensor(xh)
y = GPUArray(xh.shape, dtype=dt)

mean = GPUTensor(np.ones(1).reshape(1,1,1,1), dtype=dt)
var = GPUTensor(np.ones(1).reshape(1,1,1,1), dtype=dt)
x_desc = x.get_cudnn_tensor_desc()
print(x_desc)
print(y_desc)

param_desc = var.get_cudnn_tensor_desc()
print(param_desc)
