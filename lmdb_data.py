
import lmdb
import math
import struct
import numpy as np
from PIL import Image
from io import BytesIO

class LMDB_Data:
    def __init__(self, db_path):

        self.env = lmdb.open(db_path)
        self.txn = self.env.begin()
        self.cursor = self.txn.cursor()
        self.cursor.first()

    def num_items(self):
        return self.env.stat()['entries']

    def crop_center(self, img, crop_width, crop_height):
        
        start_x = math.ceil((img.size[0] - crop_width + 1) / 2) - 1
        start_y = math.ceil((img.size[1] - crop_height + 1) / 2) - 1

        cropped = img.crop((start_x, start_y, start_x + crop_width, start_y + crop_height))

        # cropped.save("cropped.ppm")
        return cropped

    def get_raw_item(self):
        key,val = self.cursor.item()
        self.cursor.next()
        label_len = struct.unpack("!I", val[0:4])[0]
        label = (val[4:4+label_len]).decode()
        return label, val[4+label_len:]

    def get_item(self):
        # key,val = self.cursor.item()
        # self.cursor.next()
        # label_len = struct.unpack("!I", val[0:4])[0]
        # label = (val[4:4+label_len]).decode()
        # print(key, label, len(val))
        label, data = self.get_raw_item()

        img = Image.open(BytesIO(val[4+label_len:]))
        img = self.crop_center(img, 224, 224)
        # img.save("test.ppm")
        # print(img.size)

        a = np.asarray(img)
        # print(a.shape, a.dtype)
        return label, a

if __name__ == "__main__":

    data = LMDB_Data("/tmp/lmdb/val/")

    print(data.num_items())

    label, x = data.get_item()
    print(label, x.dtype, x.shape)
