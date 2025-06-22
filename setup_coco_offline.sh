#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Bash script to perform "Option 1":
# Manual COCO and Karpathy archive extraction for HuggingFace offline mode
# ----------------------------------------------------------------------------

# 1) Download (resume-capable, skip cert checks)
wget -c --no-check-certificate https://images.cocodataset.org/zips/train2014.zip \
     -P "$HOME/.cache/huggingface/datasets/downloads"
wget -c --no-check-certificate https://images.cocodataset.org/zips/val2014.zip \
     -P "$HOME/.cache/huggingface/datasets/downloads"
wget -c --no-check-certificate https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip \
     -P "$HOME/.cache/huggingface/datasets/downloads"

# 2) Locate the HuggingFace COCO loader module directory
MODULE_DIR=$(find "$HOME/.cache/huggingface/modules/datasets_modules" -type d -name "HuggingFaceM4--COCO*" | head -1)
echo "Using COCO module dir: $MODULE_DIR"

# 3) Create expected directory structure
mkdir -p \
  "$MODULE_DIR/images/train2014" \
  "$MODULE_DIR/images/val2014" \
  "$MODULE_DIR/karpathy"

# 4) Extract downloaded zips into place
unzip -o "$HOME/.cache/huggingface/datasets/downloads/train2014.zip"   -d "$MODULE_DIR/images/train2014"
unzip -o "$HOME/.cache/huggingface/datasets/downloads/val2014.zip"     -d "$MODULE_DIR/images/val2014"
unzip -o "$HOME/.cache/huggingface/datasets/downloads/caption_datasets.zip" -d "$MODULE_DIR/karpathy"

# 5) Flatten Karpathy folder (some zips create a subdir)
if [ -d "$MODULE_DIR/karpathy/caption_datasets" ]; then
  mv "$MODULE_DIR/karpathy/caption_datasets/"* "$MODULE_DIR/karpathy/"
  rm -rf "$MODULE_DIR/karpathy/caption_datasets"
fi

# 6) Sanity checks
echo "Train images sample:"
ls "$MODULE_DIR/images/train2014" | head -10

echo "Val images sample:"
ls "$MODULE_DIR/images/val2014" | head -10

echo "Karpathy JSON present:" 
[ -f "$MODULE_DIR/karpathy/dataset_coco.json" ] && echo "OK" || echo "dataset_coco.json missing"

# 7) Export offline environment variables
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export HUGGINGFACE_HUB_OFFLINE=1

echo "Offline preparation complete. You can now run:"
echo "  python3 coco_analysis_spark_wandb.py --max_rows 100"
