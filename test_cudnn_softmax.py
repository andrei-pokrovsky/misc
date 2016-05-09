
import numpy as np
import pycuda.autoinit
import libcudnn, ctypes
from gputensor import GPUTensor
from scipy.misc import logsumexp

xo = np.array([1,2,3,4,5,6,7,8], dtype=np.float32)
print(np.log(np.exp(xo) / np.sum(np.exp(xo))))
xn = xo.reshape((1,8,1,1))

print("LOGSUMEXP", logsumexp(xn))
print(xn.shape)
x = GPUTensor(xn)
y = GPUTensor((1,8,1,1), dtype=np.float32)
print(x.shape, x.dtype)
print(y.shape, y.dtype)

cudnn_context = libcudnn.cudnnCreate()

x_desc = x.get_cudnn_tensor_desc()
y_desc = y.get_cudnn_tensor_desc()

# print(libcudnn.cudnnGetTensor4dDescriptor(x_desc))
# exit(0)

algo = libcudnn.cudnnSoftmaxAlgorithm["CUDNN_SOFTMAX_LOG"]
mode = libcudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_CHANNEL']
# mode = libcudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_INSTANCE']

alpha = 1.0
beta = 0.0
libcudnn.cudnnSoftmaxForward(cudnn_context, algo, mode, alpha, x_desc, x.get_gpu_voidp(),
        beta, x_desc, x.get_gpu_voidp())

print(x.get())

