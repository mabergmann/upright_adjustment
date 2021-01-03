import cv2
import numpy as np
import pathlib as pl
import torch
from tqdm import tqdm

from lib import utils
from lib.dataset import SUN360
from lib.metrics import Metrics
from lib.model import get_model
from lib import vis


def main():
    model = get_model()
    test_dataset = SUN360("data/test", augmentation=False)
    metric_accumulator = Metrics()
    loader = torch.utils.data.DataLoader(test_dataset, 32, shuffle=False, num_workers=16, drop_last=False,
                                worker_init_fn=utils.init_loader_seed)
    output_root = pl.Path("../output/test")

    output_root.mkdir(parents=True, exist_ok=True)
    model_path = "logs" + "/best.th"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.load_model_with_meta(model, model_path, device)
    model.eval()

    for image_th, gt in tqdm(loader):
        image_th = image_th.to(device)
        gt = gt.to(device)
        with torch.no_grad():
            output = model(image_th)
        labels = output

        # for img, lbl in zip(image_th, labels):
        #     img = vis.tensor_to_image(img)
        #     print(lbl)
        #     cv2.imshow("", img)
        #     cv2.waitKey(0)

        metric_accumulator.update(gt, labels)

    metric_accumulator.pretty_print("", None)
    metric_accumulator.plot_errors()


if __name__ == '__main__':
    main()
