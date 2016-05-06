
import enum
import math
import os.path
import numpy as np
import argparse
import json

import lmdb_data

from pycuda import gpuarray
import pycuda.autoinit

parser = argparse.ArgumentParser(description='Postprocess hypercap runs')

parser.add_argument("--model", metavar="<filename>", required=True, type=str,
                    help="json model filename")
parser.add_argument("--data", metavar="<path>", required=True, type=str,
                    help="path to lmdb dir or image directory")

args = parser.parse_args()

class GPUTensor(gpuarray.GPUArray):
    def __init__(self, initializer, dtype=np.float32):

        if isinstance(initializer, str):
            npdata = self.load_data(initializer)
            # print(npdata.shape)
            super().__init__(npdata.shape, dtype=npdata.dtype)
            self.set(npdata)
        elif isinstance(initializer, tuple):
            # print("GPUTensor(shape=", initializer)
            super().__init__(initializer, dtype=dtype)

    def load_data(self, filename):

        ext = os.path.splitext(filename)[1]
        # print(ext)
        if ext == ".npy":
            return np.load(filename)
        else:
            raise RuntimeError("Unknown tensor file extension '%s'" % ext) 

class Layer:
    def __init__(self, name=None):
        self.name = name

    def configure(self, input):
        pass

    def fprop(self, inputs, inference=False):
        raise NotImplementedError


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
    def __init__(self, config, name="Convolution"):
        super().__init__(config, name)
        self.output = None

        self.W = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][0]))
        self.bias = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][1]))
        print(self.W.shape)

    def configure(self, input):
        print("Convolution::configure: input shape =", input.shape)
        
        in_images = input.shape[0]
        in_channels = input.shape[1]
        in_height = input.shape[2]
        in_width = input.shape[3]

        filter_maps = self.W.shape[0]
        filter_channels = self.W.shape[1]
        assert(in_channels == filter_channels)
       
        out_width  = int((1.0 * in_width + 2*self.padW - self.kW) / self.dW + 1);
        out_height = int((1.0 * in_height + 2*self.padH - self.kH) / self.dH + 1);

        self.output = GPUTensor((in_images, filter_maps, out_height, out_width))
        print("Convolution::configure: output shape =", self.output.shape)
    

    def __str__(self):
        return SlidingLayer.__str__(self) + ", " + str(self.W.shape)

class Pooling(SlidingLayer):
    class Mode(enum.IntEnum):
        MAX = 1,
        AVG = 2

    def __init__(self, mode, config, name="Pooling"):
        super().__init__(config, name)
        self.mode = mode
        
        assert(config["ceil_mode"] == False)


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

class Activation(Layer):
    class Func(enum.IntEnum):
        ReLU = 1,
        TanH = 2

    def __init__(self, function):
        self.func = function

    def configure(self, input):
        self.output = input

    def __str__(self):
        return "Activation: " + self.func.name 

class Dropout(Layer):
    def __init__(self, p):
        super().__init__("Dropout")
        self.p = p

    def configure(self, input):
        self.output = input
    def __str__(self):
        return "Dropout: p=%f" % self.p

class Linear(Layer):
    def __init__(self, config):
        super().__init__("Linear")

        self.W = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][0]))
        self.bias = GPUTensor(os.path.join(config["baseDir"], config["parameterFiles"][1]))
        print(self.W.shape)

    def configure(self, input):
        print("Linear::configure: input shape =", input.shape)
        print("Linear::configure: W shape =", self.W.shape)
        print("Linear::configure: b shape =", self.bias.shape)


        
    def __str__(self):
        return "Linear: "

class SoftMax(Layer):
    class Mode(enum.IntEnum):
        FAST = 1,
        LOG = 2

    def __init__(self, mode):
        self.mode = mode

    def __str__(self):
        return "SoftMax: %s" % self.mode

class Model:
    def __init__(self, json_model_file):
        self.layers = []
        
        self.input = None

        with open(json_model_file) as f:
            jm = json.load(f)

        self.name = jm["modelName"]

        for layer in jm["layers"]:
            if layer["type"] == "View":
                continue
            layer["baseDir"] = os.path.dirname(json_model_file)
            self.layers.append(self.instantiate_layer(layer))

        # print(json.dumps(jm["layers"], indent=2))

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
        self.input = input

        if not self.layers:
            return

        self.layers[0].configure(self.input)
        for i in range(1, len(self.layers)):
            self.layers[i].configure(self.layers[i-1].output)


    def evaluate(self, input):
        self.configure(input)

# if __name__ == "__main__":
if True:

    datasrc = lmdb_data.LMDB_Data(args.data)
    print("Numer of data items: %d" % datasrc.num_items())

    yt, data = datasrc.get_item()
    model = Model(args.model)

    data = np.expand_dims(np.rollaxis(data,2), 0)
    print(model)
    y = model.evaluate(data)
