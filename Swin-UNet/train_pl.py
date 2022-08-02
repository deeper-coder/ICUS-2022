import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import utils.metrics as metrics_
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from torch.nn.modules.loss import CrossEntropyLoss

from utils.custom_loss.dice_score import dice_loss
from networks.vision_transformer import SwinUnet as ViT_seg
from datasets.dataset_spine import Spine_Dataset
from config import get_config
from parser_config import get_parser

class YasuoModel(pl.LightningModule):
    def __init__(self, config, args, net, threshold=0.5, **kwargs):
        super().__init__()
        self.config = config
        self.args = args
        self.model = net
        self.loss_fn1 = dice_loss()
        self.loss_fn2 = CrossEntropyLoss()
        self.threshold = threshold
        self.base_lr = args.base_lr

    def forward(self, img):
        output = self.model(img)
        output = F.sigmoid(output)
        return output

    def configure_optimizers(self):        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.base_lr, weight_decay=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=250, T_mult=2, eta_min=0, last_epoch=-1
        )
        
        return [optimizer], [scheduler]

    def shared_step(self, batch, stage):

        img = batch["img"]
        assert img.ndim == 4

        h, w = img.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        gt = batch["label"]
        assert gt.ndim == 4

        output = self.forward(img)
        
        # import pdb; pdb.set_trace()

        loss = self.loss_fn1(output, gt) 
        # + 0.4 * self.loss_fn2(output, gt)

        pred_mask = (output > self.threshold).type(torch.uint8)

        tp, fp, fn, tn = metrics_.get_stats(
            pred_mask.long(), gt.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        rec = metrics_.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        prec = metrics_.precision(tp, fp, fn, tn, reduction="micro-imagewise")
        f1 = metrics_.f1_score(tp, fp, fn, tn, "micro-imagewise")

        metrics = {
            # f"{stage}_per_image_iou": per_image_iou,
            # f"{stage}_dataset_iou": dataset_iou,
            f"{stage}rec": rec,
            f"{stage}prec": prec,
            f"{stage}f1": f1,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        # return self.shared_epoch_end(outputs, "train")
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
    
    args = get_parser()
    config = get_config(args)
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    images_dir = args.root_path + 'imgs/'
    labels_dir = args.root_path + 'masks/'

    # 1. create dataset
    dataset = Spine_Dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        augmentation=True,
    )

    # 2. split dataset
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. generate dataloader
    loader_args = dict(batch_size=args.batch_size, pin_memory=True)
    
    # type: ignore os.cpu_count()
    train_dataloader = DataLoader(
        dataset=train_set, shuffle=True, **loader_args)
    val_dataloader = DataLoader(dataset=val_set, shuffle=False, **loader_args)

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)
    net.load_from(config)
    # 4. create a model
    model = YasuoModel(config, args, net)

    # 5. define a trainer
    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs)

    # 6. train the network
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
