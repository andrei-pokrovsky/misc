
from scikits.cuda import cublas as cubla
import libcudnn

cublas = cubla.cublasCreate()
cudnn = libcudnn.cudnnCreate()

print("CUDNN Version: %d" % libcudnn.cudnnGetVersion())
print("CUBLAS Version:", cubla.cublasGetVersion(cublas))
