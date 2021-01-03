import matplotlib.pyplot as plt

import numpy as np
import torch


class Metrics:
    """
    Accumulate metrics during a whole train or validation pass.
    """

    def __init__(self):
        self.total_error = 0
        self.total_samples = 0
        self.cos = torch.nn.CosineSimilarity()
        self.errors = []

    def update(self, y_true, y_pred):
        """
        Updates the confusion matrix
        :param y_true: Batch containing the ground truth labels
        :param y_pred: Batch containing the predicted labels
        :return: None
        """

        cossines = self.cos(y_pred, y_true)
        angles = torch.acos(cossines)
        angles = angles * 180 / np.pi

        angles.cpu().detach().numpy()

        for error_deg in angles:
            self.total_error += error_deg
            self.total_samples += 1

            self.errors.append(error_deg)

    def get_angular_error(self):
        """
        Calculates the average angular error since last reset
        :return: Value between [0, 180] with the pixelwise accuracy
        """

        return self.total_error / self.total_samples

    def reset(self):
        """
        Resets the metric accumulator. Should be called between epochs and
        between train and validation
        :return: None
        """
        self.total_samples = 0
        self.total_error = 0
        self.errors = []

    def pretty_print(self, header, metrics=None):
        """
        Prints all the metrics
        :param header: Message printed before the metrics
        :param metrics: Which metrics should be printed
        :return: None
        """
        if metrics is None:
            metrics = ["angular_error"]
        print(header)

        for m in metrics:
            if m == 'angular_error':
                print(f"Angular Error: {self.get_angular_error()}")

    def plot_errors(self):
        y = np.asarray(range(1, len(self.errors) + 1)) * 100 / len(self.errors)
        x = sorted(self.errors)
        plt.xlim(0, 15)
        plt.plot(x, y)
        plt.show()
