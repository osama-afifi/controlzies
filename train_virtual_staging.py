import torch
import torchvision

import json
import cv2
import numpy as np
import random

from cldm.model import create_model
from share import *

import os
import sys

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse


DATA_DIR = "./training/staging_data"
PAIRED_DATA_DIR = os.path.join(
    DATA_DIR, "500_empty_staged_384px/500_empty_staged_384px"
)
UNPAIRED_DATA_DIR = os.path.join(DATA_DIR, "unpaired_384px/unpaired_384px")

STAGED = "staged"
EMPTY = "empty"
AGNOSTIC = "agnostic"
MASK = "mask"
CAPTION = "caption"
CAPTION_STAGED = f"{CAPTION}_{STAGED}"
CAPTION_EMPTY = f"{CAPTION}_{EMPTY}"

PNG = "png"


class PairedDataset(Dataset):
    """Super basic dataset with no augmentations."""

    def __init__(
        self, source_key=AGNOSTIC, target_key=STAGED, caption_key=CAPTION_STAGED
    ):
        # These keys define the task we want to train.
        self.source_key = source_key
        self.target_key = target_key
        self.caption_key = caption_key
        self.data = []
        with open(os.path.join(DATA_DIR, "paired_train_datalist.json"), "rb") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item[self.source_key]
        target_filename = item[self.target_key]
        prompt = item[self.caption_key]

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_LINEAR)
        source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_LINEAR)

        assert source.shape == target.shape, str(idx)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


def main(args):
    # Configs
    # We initialize with the pretrained ControlNet checkpoint s.t. the
    # model already knows to use the "RGB" information in the control input.
    resume_path = args.resume_path
    batch_size = 4
    logger_freq = 100
    learning_rate = 5e-6
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    staging_model = create_model("./models/control_v11p_sd15_inpaint.yaml").cpu()
    staging_model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    staging_model.learning_rate = learning_rate
    staging_model.sd_locked = sd_locked
    staging_model.only_mid_control = only_mid_control

    if args.mode == "masked_to_staged":
        dataset = PairedDataset(
            source_key=AGNOSTIC, target_key=STAGED, caption_key=CAPTION_STAGED
        )
    elif args.mode == "empty_to_staged":
        dataset = PairedDataset(
            source_key=EMPTY, target_key=STAGED, caption_key=CAPTION_STAGED
        )

    dataloader = DataLoader(
        dataset, num_workers=32, pin_memory=True, batch_size=batch_size, shuffle=True
    )
    logger = ImageLogger(batch_frequency=logger_freq)

    # Train
    trainer = pl.Trainer(
        enable_checkpointing=True, gpus=2, precision=32, callbacks=[logger]
    )
    trainer.fit(staging_model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["masked_to_staged", "empty_to_staged"]
    )
    parser.add_argument("--resume_path", type=str, required=True)  # example './models/control_v11p_sd15_inpaint.yaml'
    args = parser.parse_args()
    main(args)
