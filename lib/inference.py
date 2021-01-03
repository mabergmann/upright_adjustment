import cv2
import numpy as np
import torch

from .model import get_model
from . import utils


class Inference:
    def __init__(self, weights_path):
        self.model = get_model(pretrained=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        utils.load_model_with_meta(self.model, weights_path, self.device)

    def run_on_image(self, image):
        return self.run_on_image_list([image])[0]

    def run_on_image_list(self, image_list):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        resized_image_list = [cv2.resize(i, (224, 112)) for i in image_list]
        resized_image_list = np.asarray(resized_image_list).astype(np.float32)
        resized_image_list = resized_image_list[..., ::-1]

        batch = torch.from_numpy(resized_image_list.copy())
        batch = batch.permute(0, 3, 1, 2)
        batch /= 255

        for i in range(3):
            batch[:, i, :, :] -= mean[i]
            batch[:, i, :, :] /= std[i]

        batch = batch.to(self.device)

        with torch.no_grad():
            output = self.model(batch)

        labels = output.cpu().detach().numpy()
        labels = np.argmax(labels, axis=1)

        return list(labels)
