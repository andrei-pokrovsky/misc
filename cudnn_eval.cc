
#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

#include "toolbox/json_utils.h"
#include "error_util.h"


class GPUTensor
{

};


class Layer
{

};

class SpatialConvolution : public Layer
{
public:
    SpatialConvolution(const Json::Value &cfg)
    {
    }
};

class ReLU : public Layer
{
public:
    ReLU(const Json::Value &cfg)
    {
    }

};

class BatchNormalization : public Layer
{
public:
    BatchNormalization(const Json::Value &cfg)
    {
    }

};

class Linear: public Layer
{
public:
    Linear(const Json::Value &cfg)
    {
    }

};

class SoftMax: public Layer
{
public:
    SoftMax(const Json::Value &cfg)
    {
    }

};



class Model
{
public:
    Model(const std::string & filename)
    {
        Json::Value model = jsonutil::loadFromFile(filename);

        for (Json::Value & layer : model)
        {
            auto type = layer["type"].asString(); 
            printf("Type: %s\n", type.c_str());
            
            if (type == "SpatialConvolution")
                _layers.push_back(new SpatialConvolution(layer));
            else if (type == "ReLU")
                _layers.push_back(new ReLU(layer));
            else if (type == "SpatialBatchNormalization")
                _layers.push_back(new BatchNormalization(layer));
            else if (type == "View")
                continue;
                //_layers.push_back(new View());
            else if (type == "Linear")
                _layers.push_back(new Linear(layer));
            else if (type == "LogSoftMax")
                _layers.push_back(new SoftMax(layer));
            else
            {
                std::cerr << "ERROR. unsupported layer '" << type << "'\n'";
                exit(0);
                //throw std::runtime_error("Failed to ");
            }
        }
    }
private:
    std::vector<Layer *> _layers;

};

int main(int argc, char *argv[])
{   
    int version = (int)cudnnGetVersion();
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
    //showDevices();
    
    Model model("models/decompme/model.json");

    return 0;
}
