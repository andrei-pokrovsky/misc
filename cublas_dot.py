from scikits.cuda import cublas

cbh = cublas.cublasCreate()
print("CUBLAS Version:", cublas.cublasGetVersion(cbh))

def cublas_gemm(a, b, c):

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
