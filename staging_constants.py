# Constants
import os

DATA_DIR = "./training/staging_data"
PAIRED_DATA_DIR = os.path.join(DATA_DIR, "500_empty_staged_384px/500_empty_staged_384px")
UNPAIRED_DATA_DIR = os.path.join(DATA_DIR, "unpaired_384px/unpaired_384px")

STAGED = "staged"
EMPTY = "empty"
AGNOSTIC = "agnostic"
MASK = "mask"
CAPTION = "caption"
CAPTION_STAGED = f"{CAPTION}_{STAGED}"
CAPTION_EMPTY = f"{CAPTION}_{EMPTY}"

PNG = "png"
