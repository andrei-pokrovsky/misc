
import numpy as np
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.autoinit import context
from scikits.cuda import cublas

import cublas_dot
print(context.get_device().name())

cbh = cublas.cublasCreate()
print("CUBLAS Version:", cublas.cublasGetVersion(cbh))

def gpu_tensor_gemm(handle, a, b):
    c = gpuarray.GPUArray((a.shape[0], b.shape[1]), dtype=a.dtype)
    cublas_dot.cublas_gemm(handle, a, b, c)
    return c

for i in range(100):

    m = np.random.randint(1,1024)
    k = np.random.randint(1,1024)
    n = np.random.randint(1,1024)

    ah = np.random.rand(m, k).astype(np.float32)
    bh = np.random.rand(k, n).astype(np.float32)

    # an = np.array([[1,2],[3,4]], dtype=np.float32)
    # bn = np.array([[1,3],[2,1]], dtype=np.float32)
    ch = np.dot(ah, bh)
    # print("C:\n", ch)

    a = gpuarray.to_gpu(ah)
    b = gpuarray.to_gpu(bh)
    c = gpu_tensor_gemm(cbh, a, b)

    ch2 = c.get()
    # print(ch2)
    eq = np.allclose(ch, ch2)
    if not eq:
        print("%dx%d * %dx%d => %dx%d : %s" % (m, k, k, n, m, n, eq)) 
        print("C1 =", ch)
        print("C2 =", ch2)

cublas.cublasDestroy(cbh)


    # print("A shape =", a.shape)
# print("B shape =", b.shape)
# print("C shape =", c.shape)

# c.fill(0.0)
# print(c.get())
# print(c)
# m = a.shape[0]
# n = b.shape[1]
# k = a.shape[1]

# print("m =", m)
# print("n =", n)
# print("k =", k)
# print("a strides =", a.strides)
# print("b strides =", b.strides)
# print("c strides =", c.strides)
# lda = max([x >> 2 for x in a.strides])
# ldb = max([x >> 2 for x in b.strides])
# ldc = max([x >> 2 for x in c.strides])

# print("lda =", lda)
# print("ldb =", ldb)
# print("ldc =", ldc)

# opa = 'n'
# opb = 'n'

# print(cublas.cublasSgemm(cbh, opb, opa, n, m, k, 1.0, b.gpudata, ldb, a.gpudata, lda, 0.0, c.gpudata, ldc))

# ch = c.get()
# print(ch)
