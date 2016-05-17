
import math
import numpy as np
import pycuda.autoinit
import libcudnn
from gputensor import GPUTensor

dt = np.float16 

xh = np.ones((1,1,4,4), dtype=dt) * 2.0
# print(xh)

cudnn_context = libcudnn.cudnnCreate()

print("CUDNN Version: %d" % libcudnn.cudnnGetVersion())

x = GPUTensor(xh)
y = GPUTensor(xh.shape, dtype=dt)

pdt = np.float32

w = GPUTensor(np.ones(1).reshape(1,1,1,1), dtype=pdt)
bias = GPUTensor(np.zeros(1).reshape(1,1,1,1), dtype=pdt)
mean = GPUTensor(np.ones(1).reshape(1,1,1,1), dtype=pdt)
var = GPUTensor(np.ones(1).reshape(1,1,1,1) * 0.5, dtype=pdt)
x_desc = x.get_cudnn_tensor_desc()
y_desc = y.get_cudnn_tensor_desc()
print(x_desc)
print(y_desc)

param_desc = var.get_cudnn_tensor_desc()
print(param_desc)

eps = 0.0001

s = libcudnn.cudnnBatchNormalizationForwardInference(cudnn_context,
                libcudnn.cudnnBatchNormMode['CUDNN_BATCHNORM_SPATIAL'],
                1.0, 0.0, x_desc.ptr, x.get_gpu_voidp(),
                y_desc.ptr, y.get_gpu_voidp(),
                param_desc.ptr, w.get_gpu_voidp(), bias.get_gpu_voidp(),
                mean.get_gpu_voidp(), var.get_gpu_voidp(), eps)
print(s)
print(y.get())
