import numpy as np
# from scikits.cuda import cublas
import libcublas

def cublas_gemm(cublas_handle, a, b, c=None):
    assert(len(a.shape) == 2)
    assert(len(b.shape) == 2)
    assert(len(c.shape) == 2)
    # assert(a.dtype == np.float32)
    # assert(b.dtype == np.float32)
    # assert(c.dtype == np.float32)
    assert(a.dtype in [np.float16, np.float32])
    assert(a.dtype == b.dtype)
    assert(a.dtype == c.dtype)
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    # print("m =", m)
    # print("n =", n)
    # print("k =", k)
    # print("a strides =", a.strides)
    # print("b strides =", b.strides)
    # print("c strides =", c.strides)

    lda = int(max([x for x in a.strides]) / a.dtype.itemsize)
    ldb = int(max([x for x in b.strides]) / b.dtype.itemsize)
    ldc = int(max([x for x in c.strides]) / c.dtype.itemsize)

    # print("lda =", lda)
    # print("ldb =", ldb)
    # print("ldc =", ldc)

    opa = 'n'
    opb = 'n'
    print("Hgemm:", a.dtype)
    # print("DATA:", b.ptr, a.ptr, c.ptr)
    if a.dtype == np.float32:
        libcublas.cublasSgemm(cublas_handle, opb, opa, n, m, k, 1.0, b.gpudata, ldb, a.gpudata, lda, 0.0, c.gpudata, ldc)
    else:
        # assert(False)
    
        libcublas.cublasHgemm(cublas_handle, opb, opa, n, m, k, 1.0, b.gpudata, ldb, a.gpudata, lda, 0.0, c.gpudata, ldc)

    # ch = c.get()
    # print(ch)
    # cublas.cublasDestroy(cbh)
