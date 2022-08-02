import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import segmentation_models_pytorch as smp
from datasets.spine_dataset import Spine_Dataset

from torch.utils.data import DataLoader, random_split
from configs import parse_args
from custom_loss.dice_score import dice_loss
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


class YoneModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, lr, is_fuzzy, threshold=0.5,
                 **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            activation="sigmoid",
            **kwargs,
        )

        self.loss_fn = dice_loss()
        self.is_fuzzy = is_fuzzy
        self.threshold = threshold
        self.lr = lr

    def forward(self, img):
        # img has already normalized in dataset module!
        output = self.model(img)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=0, last_epoch=-1)
        return [optimizer], [scheduler]

    def shared_step(self, batch, stage):
        img = batch["img"]
        assert img.ndim == 4

        h, w = img.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        gt = batch["mask"]
        assert gt.ndim == 4

        output = self.forward(img)

        # select the type of loss
        if self.is_fuzzy:
            weights = (gt == 1).long() * 0.9 + (gt == 0).long() * 0.1
            gt = gt.to(torch.float32)
            loss_fn = nn.BCELoss(weight=weights)
            loss = loss_fn(output, gt)
            fuzzy_pred = output
            rec, prec, f1 = self.get_fuzzy_metrics(fuzzy_pred, gt.long())
            return {
                "loss": loss,
                "rec": rec,
                "prec": prec,
                "f1": f1,
            }
        else:
            loss = self.loss_fn(output, gt)
            pred_mask = (output > self.threshold).type(torch.uint8)
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), gt.long(), mode="binary")
            return {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

    def get_fuzzy_metrics(self, pred, gt):
        b, c, w, h = pred.shape
        zero_m = torch.zeros((b, c * w * h)).cuda()

        pred = pred.reshape(b, -1)
        gt = gt.reshape(b, -1)

        tp_ = pred.min(gt)
        tp = torch.sum(tp_, dim=1, keepdim=True)

        fp_ = (pred - gt).max(zero_m)
        fp = torch.sum(fp_, dim=1, keepdim=True)

        tn_ = (1 - pred).min(1 - gt)
        tn = torch.sum(tn_, dim=1, keepdim=True)

        fn_ = (gt - pred).max(zero_m)
        fn = torch.sum(fn_, dim=1, keepdim=True)

        rec = torch.mean(tp / (tp + fn))
        prec = torch.mean(tp / (tp + fp))
        f1 = torch.mean(2 * tp / (2 * tp + fp + fn))

        return rec, prec, f1

    def shared_epoch_end(self, outputs, stage):

        if self.is_fuzzy:
            rec = cnt1 = prec = cnt2 = f1 = cnt3 = 0
            for x in outputs:
                if not torch.isnan(x["rec"]):
                    rec += x["rec"].item()
                    cnt1 += 1
                if not torch.isnan(x["prec"]):
                    prec += x["prec"].item()
                    cnt2 += 1
                if not torch.isnan(x["f1"]):
                    f1 += x["f1"].item()
                    cnt3 += 1
            rec = rec / cnt1 if cnt1 > 0 else 0
            prec = prec / cnt2 if cnt2 > 0 else 0
            f1 = f1 / cnt3 if cnt3 > 0 else 0

        else:
            # aggregate step metics
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])

            rec = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
            prec = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
            f1 = smp.metrics.f1_score(tp, fp, fn, tn, "micro")

        metrics = {
            f"{stage}rec": rec,
            f"{stage}prec": prec,
            f"{stage}f1": f1,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="spine_proj_final_results")

    args = parse_args()
    print(args)

    # 1. create dataset
    dataset = Spine_Dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        augmentation=args.aug,
    )

    # 2. split dataset
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. generate dataloader
    loader_args = dict(batch_size=args.batch_size, num_workers=4)
    train_dataloader = DataLoader(dataset=train_set, shuffle=True, **loader_args)
    val_dataloader = DataLoader(dataset=val_set, shuffle=False, **loader_args)

    # 4. create a model
    model = YoneModel(
        arch=args.arch,
        encoder_name=args.backbone,
        encoder_weights="imagenet",
        in_channels=1,
        out_classes=1,
        lr=args.learning_rate,
        is_fuzzy=args.is_fuzzy,
    )

    # 5. define a trainer
    trainer = pl.Trainer(gpus=[1], max_epochs=args.num_epoch, logger=wandb_logger)

    # 6. train the network
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
