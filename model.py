import pytorch_lightning as pl
import torch
import torchvision, os
from torch.utils.data import DataLoader
from transformers import DetrConfig, AutoModelForObjectDetection, AutoFeatureExtractor

first_class_index = 0


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(
            img_folder, "data/coco_train.json" if train else "data/coco_val.json"
        )
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target


feature_extractor = AutoFeatureExtractor.from_pretrained(
    "hustvl/yolos-small", size=512, max_size=864
)
train_dataset = CocoDetection(img_folder="", feature_extractor=feature_extractor)
val_dataset = CocoDetection(
    img_folder="", feature_extractor=feature_extractor, train=False
)
cats = train_dataset.coco.cats
id2label = {k: v["name"] for k, v in cats.items()}


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["labels"] = labels
    return batch


train_dataloader = DataLoader(
    train_dataset,
    collate_fn=collate_fn,
    batch_size=1,
    shuffle=True,
    num_workers=8,
)
val_dataloader = DataLoader(
    val_dataset,
    collate_fn=collate_fn,
    batch_size=1,
    num_workers=8,
)


class Detr(pl.LightningModule):
    def __init__(self, lr=2.5e-5, weight_decay=1e-4):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained(
            "hustvl/yolos-small", num_labels=len(id2label), ignore_mismatched_sizes=True
        )
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader
