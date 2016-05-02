
import os
import h5py
import json
import numpy
import fnmatch

inpath = "out_joseimgnet"
outpath = "decompme"

m={}
with open(os.path.join(inpath, "model.json")) as f:
    m=json.load(f)

    for l in m:
        for i,p in enumerate(l["parameterFiles"]):
            print(p)

            l["parameterFiles"][i] = os.path.splitext(p)[0] + ".h5"

    print(m)
    
with open(os.path.join(outpath, "model.json"), "w") as f:
    json.dump(m, f, indent=2)

exit(0)



for npyfile in fnmatch.filter(os.listdir(inpath), "*.npy"):

    
    h5fn = os.path.splitext(npyfile)[0] + ".h5"
    print(npyfile, h5fn)

    tensor = numpy.load(os.path.join(inpath, npyfile))
    h5f = h5py.File(os.path.join(outpath, h5fn), 'w')
    h5f.create_dataset("params", data=tensor)
    h5f.close()

