
import math
import numpy as np
import pycuda.autoinit
import libcudnn
from gputensor import GPUTensor

xh = np.array( [[ [[1,2,3],[2,3,4],[3,4,5]],
                [[1,2,3],[2,3,4],[3,4,5]],
                [[1,2,3],[2,3,4],[3,4,5]] ] ], dtype=np.float32)
print(xh.shape)
print(xh)
cudnn_context = libcudnn.cudnnCreate()

x = GPUTensor(xh)
x_desc = x.get_cudnn_tensor_desc()
print(x_desc)

print("X:\n", x.get())
b = GPUTensor(np.array([ 1, 2, 3 ],dtype=np.float32).reshape((1,3,1,1)))
b_desc = b.get_cudnn_tensor_desc()

print(x.shape)
print(b.shape)
# print(b.get())

libcudnn.cudnnAddTensor(cudnn_context, 1.0, b_desc.ptr, b.get_gpu_voidp(),
       1.0, x_desc.ptr, x.get_gpu_voidp())

print("X2:\n", x.get())
