import cv2
import numpy as np
import pathlib as pl
from scipy.spatial.transform import Rotation as R

from lib.rotation import synthesizeRotation


images_folder = pl.Path("/home/pixforce/data/upright_adjustment/SUN360_1024/test/")
output_folder = pl.Path("data/test")

output_folder.mkdir(parents=True, exist_ok=True)

ann_lines = []

all_images = list(images_folder.glob("*.jpg"))

phi = (1 + np.sqrt(5)) / 2

for i, fname in enumerate(all_images):
    img = cv2.imread(str(fname))
    rx = i * 360 / len(all_images)
    ry = (i * 180 / phi) % 180
    ry -= 90
    rx -= 180

    print(i, rx, ry)

    r = R.from_euler("zxy", [0, rx, ry], degrees=True)
    img_rotated = synthesizeRotation(img, r.as_matrix())

    # cv2.imshow("", img_rotated)
    # cv2.waitKey(0)
    cv2.imwrite(str(output_folder/fname.name), img_rotated)

    line = f"{fname.name},{rx},{ry}\n"
    ann_lines.append(line)

with open(str(output_folder/"gt.csv"), "w") as f:
    f.writelines(ann_lines)
