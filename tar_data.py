
import math
import json
import numpy as np
import os.path
import tarfile
from PIL import Image
from io import BytesIO

class TarData:
    def __init__(self, path):

        self.tf = tarfile.TarFile(path, "r") 
        
        with open(os.path.splitext(path)[0] + ".json") as f:
            self.meta = json.load(f)


    def num_items(self):
        return len(self.meta)

    def crop_center(self, img, crop_width, crop_height):
        
        start_x = math.ceil((img.size[0] - crop_width + 1) / 2) - 1
        start_y = math.ceil((img.size[1] - crop_height + 1) / 2) - 1

        cropped = img.crop((start_x, start_y, start_x + crop_width, start_y + crop_height))

        # cropped.save("cropped.ppm")
        return cropped


    def get_item(self):
        ti = self.tf.next()

        label = os.path.dirname(ti.name)
        bufio = self.tf.extractfile(ti)
        img = Image.open(bufio)
        # print(img.size)
        img = self.crop_center(img, 224, 224)
        img.save("test.ppm")
        # print(img.size)

        a = np.asarray(img)
        # print(a.shape, a.dtype)
        return label, a

if __name__ == "__main__":

    data = TarData("/mnt/ssd/data/imagenet/val256.tar")

    print(data.num_items())

    label, x = data.get_item()
    print(label, x.dtype, x.shape)
