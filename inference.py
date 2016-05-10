
import enum
import math
import os.path
import numpy as np
import argparse
import json
import lmdb_data

import pycuda.autoinit
import pycuda.driver as drv
import context
import cublas_dot
import libcudnn, ctypes

from pycuda import gpuarray
from gputensor import GPUTensor
# import test_cublas

parser = argparse.ArgumentParser(description='Postprocess hypercap runs')

parser.add_argument("--model", metavar="<filename>", required=True, type=str,
                    help="json model filename")
parser.add_argument("--data", metavar="<path>", required=True, type=str,
                    help="path to lmdb dir or image directory")

args = parser.parse_args()

tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']


class Layer:
    def __init__(self, name=None):
        self.name = name

    def configure(self, input):
        pass

    def fprop(self, inputs, inference=False):
        raise NotImplementedError

    def __str__(self):
        return "Layer"

class SlidingLayer:
    def __init__(self, config, name=None):
        self.name = name

        for attr in [ "kW", "kH", "dH", "dW", "padH", "padW" ]:
            self.__dict__[attr] = config[attr]

    def configure(self, input):
        pass

    def fprop(self, inputs, inference=False):
        raise NotImplementedError

    def __str__(self):
        return "%s: size=%dx%d, step=%d,%d, pad=%d,%d" % (self.name, 
                self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)


class Convolution(SlidingLayer):

    # convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
    convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CONVOLUTION']
    convolution_fwd_pref = libcudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST']

    def __init__(self, config, name="Convolution"):
        super().__init__(config, name)
        self.output = None

        self.W = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][0]))

        self.alpha = 1.0
        self.beta = 0.0

        self.in_desc = None
        self.out_desc = None

        self.num_filter_maps = self.W.shape[0]
        self.num_filter_channels = self.W.shape[1]

        self.bias = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][1]), 
                shape=(1, self.num_filter_maps, 1, 1))

        # assert(self.bias.shape[0] == self.num_filter_maps)
        # self.bias = self.bias.reshape((1, self.num_filter_maps, 1, 1))
        print(self.bias.shape)
        self.b_desc = self.bias.get_cudnn_tensor_desc()

        self.filt_desc = libcudnn.cudnnCreateFilterDescriptor()
        libcudnn.cudnnSetFilter4dDescriptor(self.filt_desc, data_type, self.num_filter_maps,
                self.num_filter_channels, self.kH, self.kW)

        print("B:", self.bias.shape)
        # self.bias_desc = 
        self.conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
        libcudnn.cudnnSetConvolution2dDescriptor(self.conv_desc, self.padH, self.padW,
                self.dH, self.dW, 1, 1, self.convolution_mode)

    def __del__(self):
        if self.filt_desc:
            libcudnn.cudnnDestroyFilterDescriptor(self.filt_desc)
        if self.conv_desc:
            libcudnn.cudnnDestroyConvolutionDescriptor(self.conv_desc)

    def configure(self, input):
        print("Convolution::configure: input shape =", input.shape)
        
        in_images = input.shape[0]
        in_channels = input.shape[1]
        in_height = input.shape[2]
        in_width = input.shape[3]

        assert(in_channels == self.num_filter_channels)
       
        out_width  = int((1.0 * in_width + 2*self.padW - self.kW) / self.dW + 1);
        out_height = int((1.0 * in_height + 2*self.padH - self.kH) / self.dH + 1);

        self.output = GPUTensor((in_images, self.num_filter_maps, out_height, out_width))
        print("Convolution::configure: output shape =", self.output.shape)
   
        # initialize cudnn descriptors

        if self.in_desc:
            libcudnn.cudnnDestroyTensorDescriptor(self.in_desc.ptr)
        if self.out_desc:
            libcudnn.cudnnDestroyTensorDescriptor(self.out_desc.ptr)

        self.in_desc = input.get_cudnn_tensor_desc()
        # libcudnn.cudnnCreateTensorDescriptor()
        # libcudnn.cudnnSetTensor4dDescriptor(self.in_desc, tensor_format, data_type,
                # in_images, in_channels, in_height, in_width)


        # Get output dimensions (first two values are n_input and filters_out)
        _, _, out_height2, out_width2 = libcudnn.cudnnGetConvolution2dForwardOutputDim(
            self.conv_desc, self.in_desc.ptr, self.filt_desc)

        assert(out_width == out_width2)
        assert(out_height == out_height2)

        self.out_desc = self.output.get_cudnn_tensor_desc()
        
        # libcudnn.cudnnCreateTensorDescriptor()
        # libcudnn.cudnnSetTensor4dDescriptor(self.out_desc, tensor_format, data_type, in_images,
            # filter_maps, out_height, out_width)

        # find best convolution algorithm
        self.algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(context.cudnn, self.in_desc.ptr,
            self.filt_desc, self.conv_desc, self.out_desc.ptr, self.convolution_fwd_pref, 0)
 
        print("Convolution::configure: algo=%s" % str(self.algo))

        self.ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(context.cudnn, 
                self.in_desc.ptr, self.filt_desc, self.conv_desc, self.out_desc.ptr, self.algo)
        self.ws_ptr  = drv.mem_alloc(self.ws_size.value) if self.ws_size.value > 0 else 0

    def fprop(self, input):


        # print("\nConvolution::fprop: alpha=%f, beta=%f" % (self.alpha, self.beta))
        # print("in_data: ", input.ptr)
        # print("filt_data: ", self.W.ptr)
        # print("out_data: ", self.output.ptr)
        # print("ws_data:", self.ws_ptr, self.ws_size)
        
        ws_data = ctypes.c_void_p(int(self.ws_ptr))

        libcudnn.cudnnConvolutionForward(context.cudnn, self.alpha, 
                self.in_desc.ptr, input.get_gpu_voidp(),
                self.filt_desc, self.W.get_gpu_voidp(), 
                self.conv_desc, self.algo, ws_data, self.ws_size.value, self.beta, 
                self.out_desc.ptr, self.output.get_gpu_voidp())

        libcudnn.cudnnAddTensor(context.cudnn, 1.0, self.b_desc.ptr, self.bias.get_gpu_voidp(),
                1.0, self.out_desc.ptr, self.output.get_gpu_voidp())

    def __str__(self):
        return "%s, W=%s, b=%s" % (SlidingLayer.__str__(self), self.W.shape, self.bias.shape)


class Pooling(SlidingLayer):
    class Mode(enum.IntEnum):
        MAX = 1,
        AVG = 2

    def __init__(self, mode, config, name="Pooling"):
        super().__init__(config, name)
        self.mode = mode
        
        assert(config["ceil_mode"] == False)

        self.alpha = 1.0
        self.beta = 0.0

        self.pool_desc = None
        self.in_desc = None
        self.out_desc = None


    def configure(self, input):
        
        in_images = input.shape[0]
        in_channels = input.shape[1]
        in_height = input.shape[2]
        in_width = input.shape[3]

        assert(in_width >= self.kW)
        assert(in_height >= self.kH)

        out_width  = int((math.floor(1.0 * in_width - self.kW + 2*self.padW) / self.dW) + 1)
        out_height = int((math.floor(1.0 * in_height - self.kH + 2*self.padH) / self.dH) + 1)

        self.output = GPUTensor( (in_images, in_channels, out_height, out_width) ) 

        if self.pool_desc:
            libcudnn.cudnnDestroyPoolingDescriptor(self.pool_desc)
        if self.in_desc:
            libcudnn.cudnnDestroyTensorDescriptor(self.in_desc)
        if self.out_desc:
            libcudnn.cudnnDestroyTensorDescriptor(self.out_desc)

        self.in_desc = input.get_cudnn_tensor_desc()
        self.out_desc = self.output.get_cudnn_tensor_desc()

        self.pool_desc = libcudnn.cudnnCreatePoolingDescriptor()
        libcudnn.cudnnSetPooling2dDescriptor(self.pool_desc,
            libcudnn.cudnnPoolingMode["CUDNN_POOLING_MAX"],
            # libcudnn.cudnnNanPropagation["CUDNN_NOT_PROPAGATE_NAN"],
            self.kH, self.kW, self.padH, self.padW, self.dH, self.dW)

    def fprop(self, input):
        in_data = ctypes.c_void_p(int(input.gpudata))
        out_data = ctypes.c_void_p(int(self.output.gpudata))

        # print("Pooling::fprop()")
        # print("in_data:", input.ptr)
        # print("out_data:", self.output.ptr)

        libcudnn.cudnnPoolingForward(context.cudnn, self.pool_desc, self.alpha,
                self.in_desc.ptr, input.get_gpu_voidp(), 
                self.beta, self.out_desc.ptr, self.output.get_gpu_voidp())


class Activation(Layer):
    class Func(enum.IntEnum):
        ReLU = 1,
        TanH = 2

    def __init__(self, function):
        self.func = function
        self.alpha = 1.0
        self.beta = 0.0

    def configure(self, input):
        self.output = input

        self.inout_desc = input.get_cudnn_tensor_desc()

    def fprop(self, input):
        # print("Activation::fprop()")
        data = ctypes.c_void_p(int(input.gpudata))
        # print("data ptr =", input.ptr)
    
        libcudnn.cudnnActivationForward(context.cudnn,
                libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_RELU'],
                self.alpha,
                self.inout_desc.ptr,
                data,
                self.beta,
                self.inout_desc.ptr,
                data)

    def __str__(self):
        return "Activation: " + self.func.name 

class Dropout(Layer):
    def __init__(self, p):
        super().__init__("Dropout")
        self.p = p

    def configure(self, input):
        self.output = input

    def fprop(self, input):
        input *= self.p

    def __str__(self):
        return "Dropout: p=%f" % self.p


class Linear(Layer):
    def __init__(self, config):
        super().__init__("Linear")

        self.W = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][0]))
        self.bias = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][1]), 
                shape=(1, self.W.shape[0], 1, 1))
        # self.bias = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][1]))
        self.b_desc = self.bias.get_cudnn_tensor_desc()
        print(self.W.shape)
    
    def configure(self, input):
        print("Linear::configure: input shape =", input.shape)
        print("Linear::configure: W shape =", self.W.shape)
        print("Linear::configure: b shape =", self.bias.shape)

        elems_per_image  = np.prod(input.shape)
        # print(elems_per_image, self.W.shape[1])

        assert(elems_per_image == self.W.shape[1])
        self.output = GPUTensor((1,self.W.shape[0], 1, 1), dtype=input.dtype)
        self.output_desc = self.output.get_cudnn_tensor_desc()
        
    def fprop(self, input):
        input_2d = input.reshape((self.W.shape[1], 1)) 
        output_2d = self.output.reshape(self.W.shape[0], 1)
        # print(input_2d.flags.c_contiguous)
        # print(output_2d.flags.c_contiguous)

        # test_cublas()
        # exit(0)
        # np.save("a.npy", self.W.get())
        # np.save("b.npy", input_2d.get())

        # ad = self.W
        # print("A:", ad.shape, ad.strides, ad.size, ad.mem_size, str(ad.flags.c_contiguous))
        # print("B:", input.shape, input.strides, input.size, input.mem_size, str(input.flags.c_contiguous))
        # print("B':", input_2d.shape, input_2d.strides, input_2d.size, input_2d.mem_size, str(input_2d.flags.c_contiguous))
        # print("C:", output_2d.shape, output_2d.strides, output_2d.size, output_2d.mem_size, str(output_2d.flags.c_contiguous))
        # print("Linear::fprop()", self.W.shape, input_2d.shape, output_2d.shape)
        cublas_dot.cublas_gemm(context.cublas, self.W, input_2d, output_2d)

        # print("Linear::fprop()", self.output.shape)
        libcudnn.cudnnAddTensor(context.cudnn, 1.0, self.b_desc.ptr, self.bias.get_gpu_voidp(),
                1.0, self.output_desc.ptr, self.output.get_gpu_voidp())

    def __str__(self):
        return "Linear: %dx%d" % (self.W.shape[0], self.W.shape[1])

class SoftMax(Layer):
    class Mode(enum.IntEnum):
        FAST = 1,
        LOG = 2

    def __init__(self, mode):
        self.mode = mode

    def __str__(self):
        return "SoftMax: %s" % self.mode

    def configure(self, input):
        print("SoftMax::configure: input shape =", input.shape)

        self.in_desc = input.get_cudnn_tensor_desc()
        # self.out_desc = 
        self.output = input

    def fprop(self, input):
        algo = libcudnn.cudnnSoftmaxAlgorithm["CUDNN_SOFTMAX_LOG"]
        mode = libcudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_CHANNEL']

        alpha = 1.0
        beta = 0.0
        libcudnn.cudnnSoftmaxForward(context.cudnn, algo, mode, alpha, self.in_desc.ptr, input.get_gpu_voidp(),
                beta, self.in_desc.ptr, self.output.get_gpu_voidp())


class Model:
    def __init__(self, json_model_file):
        self.layers = []
        
        self.input = None
        self.configured_shape = None

        with open(json_model_file) as f:
            jm = json.load(f)

        self.average = jm["normalization"]["average"][0]
        self.std_dev = jm["normalization"]["stdDev"][0]
        self.name = jm["modelName"]
        self.classes = jm["classes"]

        for layer in jm["layers"]:
            if layer["type"] == "View":
                continue
            layer["baseDir"] = os.path.dirname(json_model_file)
            self.layers.append(self.instantiate_layer(layer))

        # print(json.dumps(jm["layers"], indent=2))

    def normalize(self, data):

        # print(data.shape, self.average, self.std_dev)
        data -= self.average
        data /= self.std_dev
        return data

    def instantiate_layer(self, layer):
        layer_type = layer["type"]

        if layer_type == "SpatialConvolution":
            return Convolution(layer)
        elif layer_type == "ReLU":
            return Activation(Activation.Func.ReLU)
        elif layer_type == "Threshold":
            return Activation(Activation.Func.ReLU)
        elif layer_type == "SpatialMaxPooling":
            return Pooling(Pooling.Mode.MAX, layer)
        elif layer_type == "Dropout":
            return Dropout(layer["p"])
        elif layer_type == "Linear":
            return Linear(layer)
        elif layer_type == "LogSoftMax":
            return SoftMax(SoftMax.Mode.LOG)
        else:
            raise RuntimeError("Unsupported ayer type '%s'" % layer_type)

    def __str__(self):
        s = self.name + ":\n"
        s += '\n'.join([ "   " + str(l) for l in self.layers ])
        return s

    def configure(self, input):
        print("Model::configure() input shape:", input.shape)
        self.input = input

        if not self.layers:
            return

        self.layers[0].configure(self.input)
        for i in range(1, len(self.layers)):
            self.layers[i].configure(self.layers[i-1].output)

    def evaluate(self, input):
        if self.configured_shape is None or self.configured_shape != input.shape:
            self.configure(input)
            self.configured_shape = input.shape

        self.layers[0].fprop(input)

        for i in range(1, len(self.layers)):
            self.layers[i].fprop(self.layers[i-1].output)

        y = self.layers[-1].output.get()
        i = np.argmax(y)
        return self.classes[i]


if __name__ == "__main__":
    datasrc = lmdb_data.LMDB_Data(args.data)
    print("Numer of data items: %d" % datasrc.num_items())

    model = Model(args.model)
    print(model)

    num_eq = 0
    for i in range(10000):
        yt, data = datasrc.get_item()
        data = np.ascontiguousarray(np.expand_dims(np.rollaxis(data,2), 0)).astype(np.float32)
        data = model.normalize(data)
        input_tensor = GPUTensor(data)
        # print(data.shape)
        # model.configure(input_tensor)
        y = model.evaluate(input_tensor)
        print(y, yt)
        if y == yt:
            num_eq += 1
    print("num_eq =", num_eq)
