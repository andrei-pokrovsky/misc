#!/usr/bin/env python

mport pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import libcudnn, ctypes
import numpy as np

print(drv.mem_get_info())
# Create a cuDNN context
cudnn_context = libcudnn.cudnnCreate()

# Set some options and tensor dimensions
tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
convolution_fwd_pref = libcudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST']


def benchmark_conv(kw, kh, bsz):

    start, end = (drv.Event(), drv.Event())

    def start_bench():
        start.record()

    def end_bench():
        end.record()
        end.synchronize()
        return end.time_since(start)
    n_input = bsz

    filters_in = 3
    filters_out = 64
    height_in = 224
    width_in = 224
    height_filter = kh
    width_filter = kw
    pad_h = 3
    pad_w = 3
    vertical_stride = 1
    horizontal_stride = 1
    upscalex = 1
    upscaley = 1
    alpha = 1.0
    beta = 1.0

    # Input tensor
    X = gpuarray.to_gpu(np.random.rand(n_input, filters_in, height_in, width_in)
        .astype(np.float32))

    # Filter tensor
    filters = gpuarray.to_gpu(np.random.rand(filters_out,
        filters_in, height_filter, width_filter).astype(np.float32))

    # Descriptor for input
    X_desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(X_desc, tensor_format, data_type,
        n_input, filters_in, height_in, width_in)

    # Filter descriptor
    filters_desc = libcudnn.cudnnCreateFilterDescriptor()
    libcudnn.cudnnSetFilter4dDescriptor(filters_desc, data_type, filters_out,
        filters_in, height_filter, width_filter)

    # Convolution descriptor
    conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
    libcudnn.cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w,
        vertical_stride, horizontal_stride, upscalex, upscaley,
        convolution_mode)

    # Get output dimensions (first two values are n_input and filters_out)
    _, _, height_output, width_output = libcudnn.cudnnGetConvolution2dForwardOutputDim(
        conv_desc, X_desc, filters_desc)

    # Output tensor
    Y = gpuarray.empty((n_input, filters_out, height_output, width_output), np.float32)
    y_desc = libcudnn.cudnncreatetensordescriptor()
    libcudnn.cudnnsettensor4ddescriptor(y_desc, tensor_format, data_type, n_input,
        filters_out, height_output, width_output)

    # Get pointers to GPU memory
    X_data = ctypes.c_void_p(int(X.gpudata))
    filters_data = ctypes.c_void_p(int(filters.gpudata))
    Y_data = ctypes.c_void_p(int(Y.gpudata))

    # Perform convolution
    algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(cudnn_context, X_desc,
        filters_desc, conv_desc, Y_desc, convolution_fwd_pref, 0)

    # print("Cudnn algorithm = %d" % algo.value)

    ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnn_context, X_desc, filters_desc, conv_desc, Y_desc, algo)
    ws_ptr  = drv.mem_alloc(ws_size.value) if ws_size.value > 0 else 0
    ws_data = ctypes.c_void_p(int(ws_ptr))

    libcudnn.cudnnConvolutionForward(cudnn_context, alpha, X_desc, X_data,
        filters_desc, filters_data, conv_desc, algo, ws_data, ws_size.value, beta,
        Y_desc, Y_data)
    start_bench()

    for i in range(10):
        libcudnn.cudnnConvolutionForward(cudnn_context, alpha, X_desc, X_data,
            filters_desc, filters_data, conv_desc, algo, ws_data, ws_size.value, beta,
            Y_desc, Y_data)

    ms = end_bench()

    ws_ptr = None
    libcudnn.cudnnDestroyTensorDescriptor(X_desc)
    libcudnn.cudnnDestroyTensorDescriptor(Y_desc)
    libcudnn.cudnnDestroyFilterDescriptor(filters_desc)
    libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)

    return ms / 10

# for kw in range(1, 11):
    # for kh in range(1, 11):
        # ms = benchmark_conv(kw, kh)
        # print("%dx%d : %fms" % (kw, kh, ms))
for bsz in range(1, 32):
    ms = benchmark_conv(11, 11, bsz)
    print("%d : %.2fms => %f img/sec" % (bsz, ms, bsz/ms))
# Clean up
libcudnn.cudnnDestroy(cudnn_context)
