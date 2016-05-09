
from scikits.cuda import cublas as cubla
import libcudnn

cudnn = libcudnn.cudnnCreate()
cublas = cubla.cublasCreate()

print("CUDNN Version: %d" % libcudnn.cudnnGetVersion())
print("CUBLAS Version:", cubla.cublasGetVersion(cublas))
