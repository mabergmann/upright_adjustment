import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl
from scipy.spatial.transform import Rotation as R

from lib.inference import Inference
from lib.rotation import synthesizeRotation

test_folder = pl.Path("/home/mbergmann/PycharmProjects/upright_classification/data/SUN360_1024/test")

test_images_fnames = list(test_folder.glob ("*.jpg"))

inf = Inference("logs/best.th")

errors_map = np.zeros((30, 60))

for rx in range(-180, 180, 6):
    for ry in range(-90, 90, 6):
        for n, fname in enumerate(test_images_fnames):
            r = R.from_euler("zxy", [0, rx, ry], degrees=True)
            img = cv2.imread(str(fname))
            img_rotated = synthesizeRotation(img, r.as_matrix())

            out = inf.run_on_image(img_rotated)
            expected = r.apply(np.array([0, 0, 1]))

            error = np.arccos(np.dot(out, expected))
            error = 180 * error / np.pi

            y, x = ry//6 + 15, rx//6 + 30

            errors_map[y, x] += error
            if n == 99:  # Tests 100 image in each angle
                break

        plt.imshow(errors_map)
        plt.show()

        np.save("logs/errors.npy", errors_map)
        
        



