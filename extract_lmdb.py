
import os.path
import argparse
import json
import lmdb_data
import tarfile
from io import BytesIO


parser = argparse.ArgumentParser(description='Postprocess hypercap runs')

parser.add_argument("--output", metavar="<path>", required=True, type=str,
                    help="output file base path")
parser.add_argument("--data", metavar="<path>", required=True, type=str,
                    help="path to lmdb dir or image directory")
parser.add_argument("--num-images", default=0, type=int,
                    help="number of images to evaluate, 0=all")

args = parser.parse_args()

datasrc = lmdb_data.LMDB_Data(args.data)
print("Numer of data items: %d" % datasrc.num_items())

meta = []
label_counts = {}
tf = tarfile.open(args.output + ".tar", 'w')
num = datasrc.num_items() if args.num_images == 0 else args.num_images

for i in range(num):
    label, data = datasrc.get_raw_item()
    lbc = label_counts.get(label, 0)
    img_path = os.path.join(label, "img_%04d.jpg" % lbc)
    meta.append(img_path)
    # outfile = os.path.join(args.output_dir, img_path)
    label_counts[label] = lbc + 1
    print(label, img_path, len(data))

    fo = BytesIO(data)
    ti = tarfile.TarInfo(img_path)
    ti.size = len(data)
    tf.addfile(ti, fo)

tf.close()
with open(args.output + ".json", "w") as f:
    json.dump(meta, f)
