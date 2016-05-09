import math
import numpy as np
import pycuda.autoinit
import libcudnn
from gputensor import GPUTensor

xh = np.array( [[[[ 1 + m * x for x in list(range(4)) ] for m in range(4) ]]], dtype=np.float32 )
print(xh)
print(xh.shape)

cudnn_context = libcudnn.cudnnCreate()

x = GPUTensor(xh)

x_desc = x.get_cudnn_tensor_desc()
print(x_desc)

kW = 2
kH = 2
dW = 1
dH = 1
padW = 0
padH = 0
in_width = x.shape[3]
in_height = x.shape[2]

out_width  = int((math.floor(1.0 * in_width - kW + 2*padW) / dW) + 1)
out_height = int((math.floor(1.0 * in_height - kH + 2*padH) / dH) + 1)

print("Ot:", out_width, out_height)

y = GPUTensor((1,1,out_height, out_width))
y_desc = y.get_cudnn_tensor_desc()
print(y_desc)

pool_desc = libcudnn.cudnnCreatePoolingDescriptor()
libcudnn.cudnnSetPooling2dDescriptor(pool_desc,
    libcudnn.cudnnPoolingMode["CUDNN_POOLING_MAX"],
    kH, kW, padH, padW, dH, dW)


alpha = 1.0
beta = 0.0
libcudnn.cudnnPoolingForward(cudnn_context, pool_desc, alpha,
        x_desc.ptr, x.get_gpu_voidp(), beta, y_desc.ptr, y.get_gpu_voidp())
print("SHAPE:", y_desc.shape)

print(y.get())
