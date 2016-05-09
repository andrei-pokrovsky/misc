import os.path
import numpy as np
from pycuda import gpuarray
import libcudnn, ctypes


class Tensor_Desc:
    def __init__(self, shape, dtype, fmt):
        self.desc = libcudnn.cudnnCreateTensorDescriptor()
    pass

class FilterDesc:
    pass

class PoolingDesc:
    pass


class GPUTensor(gpuarray.GPUArray):

    tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']

    def __init__(self, initializer, dtype=np.float32):

        if isinstance(initializer, str):
            npdata = self.load_data(initializer)
            # print(npdata.shape)
            super().__init__(npdata.shape, dtype=npdata.dtype)
            self.set(npdata)
        elif isinstance(initializer, tuple):
            # print("GPUTensor(shape=", initializer)
            super().__init__(initializer, dtype=dtype)
        elif isinstance(initializer, np.ndarray):
            print("SHAPE:", initializer.shape)
            super().__init__(initializer.shape, dtype=initializer.dtype)
            self.set(initializer)
        else:
            raise NotImplementedError

    def load_data(self, filename):

        ext = os.path.splitext(filename)[1]
        # print(ext)
        if ext == ".npy":
            return np.load(filename)
        else:
            raise RuntimeError("Unknown tensor file extension '%s'" % ext) 

    def get_cudnn_datatype(self):
        return libcudnn.cudnnDataType['CUDNN_DATA_FLOAT'] 

    def get_gpu_voidp(self):
        return ctypes.c_void_p(int(self.gpudata))

    def get_cudnn_tensor_desc(self):
        desc = libcudnn.cudnnCreateTensorDescriptor()
        libcudnn.cudnnSetTensor4dDescriptor(desc, self.tensor_format, self.get_cudnn_datatype(),
                self.shape[0], self.shape[1], self.shape[2], self.shape[3])
        return desc
