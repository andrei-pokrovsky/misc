
import lmdb
import math
import struct
import argparse
import torchfile
from PIL import Image
from io import BytesIO

parser = argparse.ArgumentParser(description='convert torch lmdb to portable lmdb')

parser.add_argument("--input", metavar="<path>", required=True, type=str,
                    help="input lmdb path")
parser.add_argument("--output", metavar="<path>", required=True, type=str,
                    help="output lmdb path")
parser.add_argument("--width", type=int, default=224,
                    help="width of cropped image")
parser.add_argument("--height", type=int, default=224,
                    help="height of cropped image")

args = parser.parse_args()
in_width = args.width
in_height = args.height

lmdb_in_env = lmdb.open(args.input, readonly=True)
lmdb_in_txn = lmdb_in_env.begin()
lmdb_in_cursor = lmdb_in_txn.cursor()

lmdb_out_env = lmdb.open(args.output, map_size=4*1024*1024*1024)
with lmdb_out_env.begin(write=True) as out_txn:

    count = 0
    for key, value in lmdb_in_cursor:
        # print(value)

        with open("/tmp/nasse.t7", "wb") as f:
            f.write(value)
        item = torchfile.load("/tmp/nasse.t7")

        label=item[b"Name"].decode().split('_')[0]
        # print(label)
        
        img = Image.open(BytesIO(bytes(item[b"Data"])))
        # print(img.size)

        start_x = math.ceil((img.size[0] - in_width + 1) / 2)
        start_y = math.ceil((img.size[1] - in_height + 1) / 2)

        cropped = img.crop((start_x, start_y, start_x + in_width, start_y + in_height))

        jpeg_fo = BytesIO()
        cropped.save(jpeg_fo, "jpeg")
        jpeg_fo.seek(0)
        jpeg_bytes = jpeg_fo.read()

        label_data = label.encode('ascii')

        new_data = struct.pack("!I", len(label_data)) + label_data + jpeg_bytes

        key_data = ("%07d" % count).encode('ascii')
        out_txn.put(key_data, new_data)

        count += 1
        if count % 100 == 0:
            print(count)
        # print(cropped.size)

lmdb_in_env.close()
lmdb_out_env.close()
