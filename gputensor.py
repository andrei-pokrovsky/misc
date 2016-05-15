import os.path
import numpy as np
from pycuda import gpuarray
import libcudnn, ctypes

np_2_cudnn_fmt = { 
    np.dtype(np.float16): libcudnn.cudnnDataType['CUDNN_DATA_HALF'],
    np.dtype(np.float32): libcudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
    np.dtype(np.float64): libcudnn.cudnnDataType['CUDNN_DATA_DOUBLE']
}

# print(np_2_cudnn_fmt[np.float16])
# exit(0)
class TensorDesc:
    def __init__(self, shape, dtype, fmt=libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']):
        self.ptr = libcudnn.cudnnCreateTensorDescriptor()

        libcudnn.cudnnSetTensor4dDescriptor(self.ptr, fmt, dtype,
                shape[0], shape[1], shape[2], shape[3])


    def get(self):
        return libcudnn.cudnnGetTensor4dDescriptor(self.ptr)
    def __str__(self):
        elems = self.get() 
        return "Tensor: dtype=%s, shape=(%d,%d,%d,%d), strides=(%d,%d,%d,%d)" % elems

    @property
    def shape(self):
        return self.get()[1:5]

class FilterDesc:
    pass

class PoolingDesc:
    pass


class GPUTensor(gpuarray.GPUArray):

    tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']

    def __init__(self, initializer, dtype=None, shape=None):

        if isinstance(initializer, str):
            npdata = self.load_data(initializer)
            if dtype and dtype != npdata.dtype:
                npdata = npdata.astype(dtype, copy=False)
            if shape is not None:
                npdata = npdata.reshape(shape)
            super().__init__(npdata.shape, dtype=npdata.dtype)
            self.set(npdata)
        elif isinstance(initializer, tuple):
            # print("GPUTensor(shape=", initializer)
            super().__init__(initializer, dtype=dtype)
        elif isinstance(initializer, np.ndarray):
            # print("SHAPE:", initializer.shape)
            if dtype and dtype != npdata.dtype:
                initializer = initializer.astype(dtype)

            if shape is not None and shape != initializer.shape:
                initializer = initializer.reshape(shape)
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
        print("HERE:", type(self.dtype), type(np.dtype(np.float16)))
        return np_2_cudnn_fmt[self.dtype]
        # return libcudnn.cudnnDataType['CUDNN_DATA_FLOAT'] 

    def get_gpu_voidp(self):
        return ctypes.c_void_p(int(self.gpudata))

    def get_cudnn_tensor_desc(self):
        # desc = libcudnn.cudnnCreateTensorDescriptor()
        # libcudnn.cudnnSetTensor4dDescriptor(desc, self.tensor_format, self.get_cudnn_datatype(),
                # self.shape[0], self.shape[1], self.shape[2], self.shape[3])
        # return desc
        return TensorDesc(self.shape, self.get_cudnn_datatype())
