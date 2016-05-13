
import enum
import math
import os.path
import numpy as np
import argparse
import json
import time
import tar_data

import pycuda.autoinit
import pycuda.driver as drv
import context
import cublas_dot
import libcudnn, ctypes

from pycuda import gpuarray
from gputensor import GPUTensor

parser = argparse.ArgumentParser(description='Postprocess hypercap runs')

parser.add_argument("--model", metavar="<filename>", required=True, type=str,
                    help="json model filename")
parser.add_argument("--data", metavar="<path>", required=True, type=str,
                    help="path to lmdb dir or image directory")
parser.add_argument("--precision", default="fp32", type=str, choices=["fp32","fp16"],
                    help="floating point precision to use")
parser.add_argument("--num-images", default=0, type=int,
                    help="number of images to evaluate, 0=all")
parser.add_argument("--benchmark", default=False, action='store_true', 
                    help="benchmark network with single batch")

args = parser.parse_args()

tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']

np_2_cudnn_fmt = { 
    np.float16: libcudnn.cudnnDataType['CUDNN_DATA_HALF'],
    np.float32: libcudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
    np.float64: libcudnn.cudnnDataType['CUDNN_DATA_DOUBLE']
}

class Layer:
    def __init__(self, name=None):
        self.name = name
        self.truth = None

    def configure(self, input):
        pass

    def fprop(self, inputs, inference=False):
        raise NotImplementedError

    def check_truth(self, atol=0.0005):
        if not self.truth:
            return
        truth = self.truth[0]
        output = self.output[0]
        if output.shape != truth.shape:
            output.reshape(truth.shape)

        if not np.allclose(truth, output, atol=atol):
            print("%s COMPARE FAILED:" % self.name)
            print(truth.shape)
            print(output.shape)
            print(truth[0][0])
            print(output[0][0])
            print("MAX DIFF:", np.max(np.abs(truth - output)))
            exit(1)
        else:
            print("%s COMPARED OK" % self.name)
    
    def load_tensor(self, config, index):
        return GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][index]))


    def __str__(self):
        return "Layer"


class SlidingLayer(Layer):
    def __init__(self, config, name=None):
        super().__init__(name)

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

    convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
    # convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CONVOLUTION']
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
        # print(self.bias.shape)
        self.b_desc = self.bias.get_cudnn_tensor_desc()

        self.filt_desc = libcudnn.cudnnCreateFilterDescriptor()
        libcudnn.cudnnSetFilter4dDescriptor(self.filt_desc, data_type, self.num_filter_maps,
                self.num_filter_channels, self.kH, self.kW)

        # print("B:", self.bias.shape)
        # self.bias_desc = 
        self.conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
        libcudnn.cudnnSetConvolution2dDescriptor(self.conv_desc, self.padH, self.padW,
                self.dH, self.dW, 1, 1, self.convolution_mode)

    def __del__(self):
        pass
        # if self.filt_desc:
            # libcudnn.cudnnDestroyFilterDescriptor(self.filt_desc)
        # if self.conv_desc:
            # libcudnn.cudnnDestroyConvolutionDescriptor(self.conv_desc)

    def configure(self, input):
        # print("Convolution::configure: input shape =", input.shape)
        
        in_images = input.shape[0]
        in_channels = input.shape[1]
        in_height = input.shape[2]
        in_width = input.shape[3]

        assert(in_channels == self.num_filter_channels)
       
        out_width  = int((1.0 * in_width + 2*self.padW - self.kW) / self.dW + 1);
        out_height = int((1.0 * in_height + 2*self.padH - self.kH) / self.dH + 1);

        self.output = GPUTensor((in_images, self.num_filter_maps, out_height, out_width))
        # print("Convolution::configure: output shape =", self.output.shape)
   
        # initialize cudnn descriptors
        if self.in_desc:
            libcudnn.cudnnDestroyTensorDescriptor(self.in_desc.ptr)
        if self.out_desc:
            libcudnn.cudnnDestroyTensorDescriptor(self.out_desc.ptr)

        self.in_desc = input.get_cudnn_tensor_desc()

        # Get output dimensions (first two values are n_input and filters_out)
        _, _, out_height2, out_width2 = libcudnn.cudnnGetConvolution2dForwardOutputDim(
            self.conv_desc, self.in_desc.ptr, self.filt_desc)

        assert(out_width == out_width2)
        assert(out_height == out_height2)

        self.out_desc = self.output.get_cudnn_tensor_desc()
        
        # find best convolution algorithm
        self.algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(context.cudnn, self.in_desc.ptr,
            self.filt_desc, self.conv_desc, self.out_desc.ptr, self.convolution_fwd_pref, 0)
 
        # print("Convolution::configure: algo=%s" % str(self.algo))

        self.ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(context.cudnn, 
                self.in_desc.ptr, self.filt_desc, self.conv_desc, self.out_desc.ptr, self.algo)
        self.ws_ptr  = drv.mem_alloc(self.ws_size.value) if self.ws_size.value > 0 else 0

    def fprop(self, input):

        # print("\nConvolution::fprop: alpha=%f, beta=%f" % (self.alpha, self.beta))
        
        ws_data = ctypes.c_void_p(int(self.ws_ptr))

        libcudnn.cudnnConvolutionForward(context.cudnn, self.alpha, 
                self.in_desc.ptr, input.get_gpu_voidp(),
                self.filt_desc, self.W.get_gpu_voidp(), 
                self.conv_desc, self.algo, ws_data, self.ws_size.value, self.beta, 
                self.out_desc.ptr, self.output.get_gpu_voidp())

        libcudnn.cudnnAddTensor(context.cudnn, 1.0, self.b_desc.ptr, self.bias.get_gpu_voidp(),
                1.0, self.out_desc.ptr, self.output.get_gpu_voidp())

        self.check_truth()

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

        self.check_truth()

class Activation(Layer):
    class Func(enum.IntEnum):
        ReLU = 1,
        TanH = 2

    def __init__(self, function):
        super().__init__(str(function))
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

        self.check_truth()

    def __str__(self):
        return "Activation: " + self.func.name 

class Dropout(Layer):
    def __init__(self, p):
        super().__init__("Dropout")
        self.p = 1.0

    def configure(self, input):
        self.output = input

    def fprop(self, input):
        input *= self.p

    def __str__(self):
        return "Dropout: p=%f" % self.p


class BatchNormalization(Layer):
    def __init__(self, config):
        super().__init__("Linear")

        assert(config["affine"])
        
        self.eps = config["eps"]

        variance = np.load(os.path.join(config["baseDir"], config["parameterFiles"][3]))
        nelem = variance.shape[0]

        if config["varianceFormat"] == "variance" and libcudnn.cudnnGetVersion() < 5000:
            # print("FIXING variance format")
            variance += self.eps
            variance = np.reciprocal(np.sqrt(variance))
            
        self.variance = GPUTensor(variance, shape=(1, nelem, 1, 1))

        self.W = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][0]), shape=(1, nelem, 1, 1))
        self.bias = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][1]), shape=(1, nelem, 1, 1)) 
                # shape=(1, self.W.shape[0], 1, 1))
        self.average = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][2]), shape=(1, nelem, 1, 1))

        self.param_desc = self.W.get_cudnn_tensor_desc()
        self.in_desc = None
        self.out_desc = None

    def configure(self, input):
        self.output = GPUTensor(input.shape)

        if self.in_desc:
            libcudnn.cudnnDestroyTensorDescriptor(self.in_desc.ptr)
        if self.out_desc:
            libcudnn.cudnnDestroyTensorDescriptor(self.out_desc.ptr)

        self.in_desc = input.get_cudnn_tensor_desc()
        self.out_desc = self.output.get_cudnn_tensor_desc()
        # print("BatchNormalization:configure() input=", input.shape, self.W.shape[0])

    def fprop(self, input):
        # The input transformation performed by this function is defined as: 
        # y := alpha*y + beta *(bnScale * (x-estimatedMean)/sqrt(epsilon + estimatedVariance)+bnBias)
        libcudnn.cudnnBatchNormalizationForwardInference(context.cudnn, 
                libcudnn.cudnnBatchNormMode['CUDNN_BATCHNORM_SPATIAL'],
                1.0, 0.0, self.in_desc.ptr, input.get_gpu_voidp(),
                self.out_desc.ptr, self.output.get_gpu_voidp(),
                self.param_desc.ptr, self.W.get_gpu_voidp(), self.bias.get_gpu_voidp(),
                self.average.get_gpu_voidp(), self.variance.get_gpu_voidp(), self.eps)

        self.check_truth()

    def __str__(self):
        return "BatchNormalization: %dx%d" % (self.W.shape[0], self.bias.shape[0])

class Linear(Layer):
    def __init__(self, config):
        super().__init__("Linear")

        self.W = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][0]))
        self.bias = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][1]), 
                shape=(1, self.W.shape[0], 1, 1))
        # self.bias = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][1]))
        self.b_desc = self.bias.get_cudnn_tensor_desc()
        # print(self.W.shape)
    
    def configure(self, input):
        # print("Linear::configure: input shape =", input.shape)
        # print("Linear::configure: W shape =", self.W.shape)
        # print("Linear::configure: b shape =", self.bias.shape)

        elems_per_image  = np.prod(input.shape)
        # print(elems_per_image, self.W.shape[1])

        assert(elems_per_image == self.W.shape[1])
        self.output = GPUTensor((1,self.W.shape[0], 1, 1), dtype=input.dtype)
        self.output_desc = self.output.get_cudnn_tensor_desc()
        
        if self.truth is not None:
            print("OUTPUT TRUTH SHAPE:", self.truth.shape, self.output.shape)

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

        self.check_truth()


    def __str__(self):
        return "Linear: %dx%d" % (self.W.shape[0], self.W.shape[1])

class SoftMax(Layer):
    class Mode(enum.IntEnum):
        FAST = 1,
        LOG = 2

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def __str__(self):
        return "SoftMax: %s" % self.mode

    def configure(self, input):
        # print("SoftMax::configure: input shape =", input.shape)

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

        self.check_truth()

class Model:
    def __init__(self, json_model_file, load_truth=False):
        self.layers = []
        
        self.input = None
        self.configured_shape = None

        with open(json_model_file) as f:
            jm = json.load(f)

        self.average = jm["normalization"]["average"][0]
        self.std_dev = jm["normalization"]["stdDev"][0]
        self.name = jm["modelName"]
        self.classes = jm["classes"]

        gi = 0
        for layer in jm["layers"]:
            if layer["type"] == "View":
                continue
            layer["baseDir"] = os.path.dirname(json_model_file)
            self.layers.append(self.instantiate_layer(layer))

            if load_truth:
                gtfn = os.path.join("truth", "layer_%02d_output.npy" % gi)
                if os.path.isfile(gtfn):
                    self.layers[-1].truth = np.load(gtfn)
                    print("Loaded truth for layer %d from %s" % (gi, gtfn))
                gi += 1

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
        elif layer_type == "SpatialBatchNormalization":
            return BatchNormalization(layer)
        elif layer_type == "Dropout":
            return Dropout(layer["p"])
        elif layer_type == "Linear":
            return Linear(layer)
        elif layer_type == "LogSoftMax":
            return SoftMax(SoftMax.Mode.LOG)
        else:
            raise RuntimeError("Unsupported layer type '%s'" % layer_type)

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

        # print("INPUT:", self.input.get()[0][1][1])
        self.layers[0].fprop(input)

        for i in range(1, len(self.layers)):
            self.layers[i].fprop(self.layers[i-1].output)

        y = self.layers[-1].output.get()
        i = np.argmax(y)
        return self.classes[i]

def benchmark(datasrc, model):
    start = time.time()
    label, data = datasrc.get_item()
    print("Data load time: %.2fms" % ((time.time() - start) * 1000.0))

    start = time.time()
    data = np.ascontiguousarray(np.expand_dims(np.rollaxis(data,2), 0)).astype(np.float32)
    data = model.normalize(data)
    print("Data prep time: %.2fms" % ((time.time() - start) * 1000.0))

    input_tensor = GPUTensor(data)
    y = model.evaluate(input_tensor)
    start = time.time()
    num_iterations = 10
    for i in range(num_iterations):
        y = model.evaluate(input_tensor)
        print(y)

    et = (time.time() - start) * 1000 / num_iterations
    print("Model eval time: %.2fms = %.1ffps" % (et, 1000.0 / et))


if __name__ == "__main__":
    datasrc = tar_data.TarData(args.data)
    print("Numer of data items: %d" % datasrc.num_items())

    # yt, data = datasrc.get_item()
    # print(data.shape)
    # exit(0)
    model = Model(args.model)
    print(model)

    if args.benchmark:
        benchmark(datasrc, model)
        exit(0)

    num_errors = 0
    num = datasrc.num_items() if args.num_images == 0 else args.num_images

    for i in range(num):
        yt, data = datasrc.get_item()
        data = np.ascontiguousarray(np.expand_dims(np.rollaxis(data,2), 0)).astype(np.float32)
        data = model.normalize(data)
        # yt = results[i][0]
        # data = np.expand_dims(inputs[i], 0)
        # print(data.shape, data.dtype)
        # print(data2.shape, data2.dtype)
        # print(np.allclose(data,data2))
        # continue
        # exit(0)
        input_tensor = GPUTensor(data)
        # print(data.shape)
        # model.configure(input_tensor)
        y = model.evaluate(input_tensor)
        print(y, yt)
        if y != yt:
            num_errors += 1
    print("DONE: %d images classified, error rate=%.4f" % (num, 1.0 * num_errors / num))
