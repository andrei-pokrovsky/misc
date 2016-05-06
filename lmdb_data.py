
import lmdb
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

    def get_item(self):
        key,val = self.cursor.item()

        label_len = struct.unpack("!I", val[0:4])[0]
        label = (val[4:4+label_len]).decode()
        print(key, label, len(val))

        img = Image.open(BytesIO(val[4+label_len:]))
        # print(img.size)

        a = np.asarray(img)
        # print(a.shape, a.dtype)
        return label, a

if __name__ == "__main__":

    data = LMDB_Data("/tmp/lmdb/val/")

    print(data.num_items())

    label, x = data.get_item()
    print(label, x.dtype, x.shape)
