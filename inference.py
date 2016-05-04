
import os.path
import numpy as np
import argparse
import json


from pycuda import gpuarray

parser = argparse.ArgumentParser(description='Postprocess hypercap runs')

parser.add_argument("--model", metavar="<filename>", required=True, type=str,
                    help="json model filename")

args = parser.parse_args()

class GPUTensor:
    def __init__(self, data_filename):

        npdata = self.load_data(data_filename)

        # self.

    def load_data(self, filename):

        ext = os.path.splitext(data_filename)[1]
        print(ext)
        if ext == ".npy":
            return npy.load(data_filename)
        else:
            raise RuntimeError("Unknown tensor file extension '%s'" % ext) 

class Layer:
    def __init__(self, name=None):
        self.name = name

    def fprop(self, inputs, inference=False):
        raise NotImplementedError

class Convolution(Layer):
    def __init__(self, config, name="Convolution"):
        super().__init__(name)


class Activation(Layer):
    ReLU = 1
    TanH = 2

    def __init__(self, t):
        self.type = t

class Pooling(Layer):
    MAX = 1
    AVG = 2

    def __init__(self, mode):
        self.mode = mode

class Dropout(Layer):
    def __init__(self, p):
        super().__init__("Linear")
        self.p = p

class Linear(Layer):
    def __init__(self, config):
        super().__init__("Linear")

class SoftMax(Layer):
    FAST = 1
    LOG = 2

    def __init__(self, mode):
        self.mode = mode

class Model:
    def __init__(self, json_model_file):
        self.layers = []
        
        with open(json_model_file) as f:
            jm = json.load(f)

        for layer in jm["layers"]:
            if layer["type"] == "View":
                continue
            self.layers.append(self.instantiate_layer(layer))

        print(json.dumps(jm["layers"], indent=2))

    def instantiate_layer(self, layer):
        layer_type = layer["type"]

        if layer_type == "SpatialConvolution":
            return Convolution(layer)
        elif layer_type == "ReLU":
            return Activation(Activation.ReLU)
        elif layer_type == "Threshold":
            return Activation(Activation.ReLU)
        elif layer_type == "SpatialMaxPooling":
            return Pooling(Pooling.MAX)
        elif layer_type == "Dropout":
            return Dropout(layer["p"])
        elif layer_type == "Linear":
            return Linear(layer)
        elif layer_type == "LogSoftMax":
            return SoftMax(SoftMax.LOG)
        else:
            raise RuntimeError("Unsupported ayer type '%s'" % layer_type)




model = Model(args.model)
