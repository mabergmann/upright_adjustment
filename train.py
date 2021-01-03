import cv2
import numpy as np
import pathlib as pl
import torch
from torch.optim import Adam
from tqdm import tqdm

from lib.model import get_model
from lib.dataset import SUN360
from lib import vis, utils
from lib.metrics import Metrics


def main():
    batch_size = 12
    batches_per_update = 4
    max_epocs = 300
    patience = 20
    log_path = pl.Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    model = get_model()
    train_dataset = SUN360("data/train", augmentation=True)
    val_dataset = SUN360("data/val", augmentation=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)

    # for img, gt in train_dataset:
    #     img = vis.tensor_to_image(img).astype("uint8")
    #     print(gt)
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)

    weights_backbone = []
    bias_backbone = []

    weights_regressor = []
    bias_regressor = []
    for n, p in model.named_parameters():
        if n.startswith("regressor") and n.endswith("weight"):
            weights_regressor.append(p)

        elif n.startswith("regressor"):
            bias_regressor.append(p)

        elif n.endswith("weight"):
            weights_backbone.append(p)

        else:
            bias_backbone.append(p)

    optimizer = Adam([
        {"params": bias_backbone, "lr": 0.001},
        {"params": weights_backbone, "lr": 0.001, "weight_decay": 0.00},
        {"params": bias_regressor, "lr": 0.01},
        {"params": weights_regressor, "lr": 0.01, "weight_decay": 0.00},
    ])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', factor=0.1,
        patience=10, verbose=True)

    criterion = torch.nn.MSELoss()

    metric_accumulator = Metrics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_epoch = -1
    best_val_metric = float("inf")

    for epoch in range(1, max_epocs + 1):
        metric_accumulator.reset()
        model.train()
        mean_loss = 0.0

        optimizer.zero_grad()

        for i_batch, (image_th, gt) in enumerate(tqdm(train_loader,
                                                      desc=f"Train epoch {epoch}")):

            image_th = image_th.to(device)
            gt = gt.to(device)
            output = model(image_th)
            total_loss = criterion(output, gt)
            total_loss = total_loss / batches_per_update

            total_loss.backward()

            if (i_batch + 1) % batches_per_update == 0:
                optimizer.step()
                optimizer.zero_grad()

            mean_loss += total_loss.cpu().detach().numpy()

            labels = output

            metric_accumulator.update(gt, labels)

        mean_loss /= len(train_loader)

        metric_accumulator.pretty_print(
            f"========== Train Metrics =========="
            f"\nLoss: {mean_loss}")
        metric_accumulator.reset()

        model.eval()

        for i_batch, (image_th, gt) in enumerate(tqdm(val_loader,
                                                      desc=f"Val epoch {epoch}")):
            image_th = image_th.to(device)
            gt = gt.to(device)
            with torch.no_grad():
                output = model(image_th)
            labels = output
            metric_accumulator.update(gt, labels)

        val_main_metric = metric_accumulator.get_angular_error()

        lr_scheduler.step(val_main_metric)

        metric_accumulator.pretty_print(
            f"========== Validation metrics =========="
            f"\nBest epoch: {best_epoch}"
            f"\nBest error: {best_val_metric} ",
            None)

        # Select best epoch and stop the training if the patience is over
        utils.save_model_with_meta(model, log_path / "last.th", optimizer,
                                   {'val_main_metric': val_main_metric})
        if best_val_metric > val_main_metric:
            best_val_metric = val_main_metric

            utils.save_model_with_meta(model, log_path / "best.th", optimizer,
                                       {'val_main_metric': val_main_metric})
            print('New best validation metric. Saving...')

            best_epoch = epoch

        if (epoch - best_epoch) > patience:
            print("Finishing train")
            break
    

if __name__ == '__main__':
    main()
