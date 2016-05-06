
import numpy as np
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.autoinit import context
from scikits.cuda import cublas

print(context.get_device().name())

cbh = cublas.cublasCreate()
print("CUBLAS Version:", cublas.cublasGetVersion(cbh))

an = np.array([[1,2],[3,4]],dtype=np.float32)
bn = np.array([[1,3],[2,1]], dtype=np.float32)

cn = np.dot(an,bn)
print("C:\n", cn)

a=gpuarray.to_gpu(an)
b=gpuarray.to_gpu(bn)
# c=gpuarray.GPUArray((1,2), dtype=np.float32) 
c=gpuarray.to_gpu(cn)
print("A shape =", a.shape)
print("B shape =", b.shape)
print("C shape =", c.shape)

c.fill(0.0)
print(c.get())
print(c)
m = a.shape[0]
n = b.shape[1]
k = a.shape[1]

print("m =", m)
print("n =", n)
print("k =", k)
print("a strides =", a.strides)
print("b strides =", b.strides)
print("c strides =", c.strides)
lda = max([x >> 2 for x in a.strides])
ldb = max([x >> 2 for x in b.strides])
ldc = max([x >> 2 for x in c.strides])

print("lda =", lda)
print("ldb =", ldb)
print("ldc =", ldc)

opa = 'n'
opb = 'n'

print(cublas.cublasSgemm(cbh, opb, opa, n, m, k, 1.0, b.gpudata, ldb, a.gpudata, lda, 0.0, c.gpudata, ldc))

ch = c.get()
print(ch)
cublas.cublasDestroy(cbh)
