# %% h
import numpy as np
import convert as via2coco
import torchvision, os
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from transformers import AutoFeatureExtractor


first_class_index = 0
# %% create dataloaders


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(
            img_folder, "custom_train.json" if train else "custom_val.json"
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
train_dataset = CocoDetection(
    img_folder="/content/balloon/train", feature_extractor=feature_extractor
)
val_dataset = CocoDetection(
    img_folder="/content/balloon/val", feature_extractor=feature_extractor, train=False
)

# %% visualize sample images

# based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
image_ids = train_dataset.coco.getImgIds()
# let's pick a random image
image_id = image_ids[np.random.randint(0, len(image_ids))]
print("Image n°{}".format(image_id))
image = train_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join("/content/balloon/train", image["file_name"]))

annotations = train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v["name"] for k, v in cats.items()}

for annotation in annotations:
    box = annotation["bbox"]
    class_idx = annotation["category_id"]
    x, y, w, h = tuple(box)
    draw.rectangle((x, y, x + w, y + h), outline="red", width=1)
    draw.text((x, y), id2label[class_idx], fill="white")

image

# %% create dataloaders with proper config


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["labels"] = labels
    return batch


train_dataloader = DataLoader(
    train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True
)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)
batch = next(iter(train_dataloader))

# %%
import pytorch_lightning as pl
from transformers import DetrConfig, AutoModelForObjectDetection
import torch


class Detr(pl.LightningModule):
    def __init__(self, lr, weight_decay):
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


# %% create model and sample inference
model = Detr(lr=2.5e-5, weight_decay=1e-4)

outputs = model(pixel_values=batch["pixel_values"])

# %%

from pytorch_lightning import Trainer

trainer = Trainer(
    gpus=1, max_steps=2000, gradient_clip_val=0.1, accumulate_grad_batches=4
)
trainer.fit(model)

# %% get ground truths

from datasets import get_coco_api_from_dataset

base_ds = get_coco_api_from_dataset(val_dataset)

# %%
from datasets.coco_eval import CocoEvaluator
from tqdm.notebook import tqdm

iou_types = ["bbox"]
coco_evaluator = CocoEvaluator(
    base_ds, iou_types
)  # initialize evaluator with ground truths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

print("Running evaluation...")

for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    labels = [
        {k: v.to(device) for k, v in t.items()} for t in batch["labels"]
    ]  # these are in DETR format, resized + normalized

    # forward pass
    outputs = model.model(pixel_values=pixel_values)

    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(
        outputs, orig_target_sizes
    )  # convert outputs of model to COCO api
    res = {target["image_id"].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()
