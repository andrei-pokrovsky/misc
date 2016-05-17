
import numpy as np
import pycuda.driver as drv
import pycuda.autoinit
from pycuda import gpuarray,driver

from gputensor import GPUTensor
import cublas_dot
import context

# print(context.get_device().name())


test_dtype = np.float16

def check_results(c, c2):
    if test_dtype == np.float32:
        atol = 0.00005 
        rtol = 1e-5
    else:
        atol = 0.005
        rtol = 0.002
    ok = np.allclose(c, c2, atol=atol, rtol=rtol)
    print("   ISOK:", ok) #, equal_nan=True))
    if not ok:
        print("C2:")
        print(c2)

        abs_diff = np.max(np.abs(c - c2))
        relmin = np.amin(c/c2)
        relmax = np.amax(c/c2)
        print("   MAX ABS DIFF: %.9f" % abs_diff)
        print("   REL DIFFS:", relmin,relmax)
        exit(0)


def gpu_tensor_gemm(handle, a, b):
    c = gpuarray.GPUArray((a.shape[0], b.shape[1]), dtype=a.dtype)
   # c.fill(0)
    # driver.memset_d16(c.gpudata, 15360, c.mem_size)
    # print(c.get())
    cublas_dot.cublas_gemm(handle, a, b, c)
    return c

def test_saved(precision):

    a = np.load("a%d.npy" % precision)
    b = np.load("b%d.npy" % precision)
    print("NANS:", np.isnan(a).any() or np.isnan(b).any())

    c = np.dot(a,b)
    print(b)
    print("A:", a.shape, a.dtype)
    print("B:", b.shape, b.dtype)
    print("C:", c.shape, c.dtype)

# ad = gpuarray.to_gpu(a)
# bd = gpuarray.to_gpu(b)
    ad = GPUTensor(a)
    bd = GPUTensor(b)
    cd = gpu_tensor_gemm(context.cublas, ad, bd)

    print("A:", ad.shape, ad.strides, ad.size, ad.mem_size, str(ad.flags.c_contiguous))
    print("B:", bd.shape, bd.strides, bd.size, bd.mem_size, str(bd.flags.c_contiguous))
    print("C:", cd.shape, cd.strides, cd.size, cd.mem_size, str(cd.flags.c_contiguous))
    c2 = cd.get()
    # print("C2:", c2.shape)
    check_results(c, c2)
    # print(c2)

def test_ranges():
        
    np.random.seed(1)
    for i in range(100):

        max_size = 512
        m = np.random.randint(1,max_size)
        k = np.random.randint(1,max_size)
        n = np.random.randint(1,max_size)

        # m = 2
        # k = 2
        # n = 2
        ah = np.random.rand(m, k).astype(test_dtype)
        bh = np.random.rand(k, n).astype(test_dtype)

        print("SIZE: ", m, n, k, "VALS:", ah[0][0], bh[0][0])
        # ah = np.array([[1,2.1],[3,4.1]], dtype=test_dtype)
        # bh = np.array([[1,3],[2,1.1]], dtype=test_dtype)
        ch = np.dot(ah, bh)
        print(ch.dtype)
        # print("C:\n", ch)

        a = gpuarray.to_gpu(ah)
        b = gpuarray.to_gpu(bh)
        c = gpu_tensor_gemm(context.cublas, a, b)

        ch2 = c.get()
        check_results(ch, ch2)
        # # print(ch2)
        # eq = np.allclose(ch, ch2, atol=0.00001)
        # print("%d,%d,%d: %s" % (n, m, k, str(eq)))
        # if not eq:
            # print("%dx%d * %dx%d => %dx%d : %s" % (m, k, k, n, m, n, eq)) 
            # print("C1 =", ch)
            # print("C2 =", ch2)

if __name__ == "__main__":
    test_ranges()
    #test_saved(32)


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
