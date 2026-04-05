#!/bin/bash
# Convert a Tanks-and-Temples scene into the ScanNet++-formatted directory structure.
#
# Usage:
#   sh data/tt/prep_tt_ph.sh <scene> <root>
#
# Example:
#   sh data/tt/prep_tt_ph.sh truck data/tt
#
# After running, the scene directory will contain:
#   <root>/datasets/<scene>/
#     colmap/
#       cameras.txt
#       cameras_fish.txt
#       images.txt
#       points3D.txt
#     resized_images/
#       000000.jpg / .png
#       000001.jpg / .png
#       …

set -e

SCENE=$1
ROOT=$2

if [ -z "$SCENE" ] || [ -z "$ROOT" ]; then
  echo "Usage: sh prep_tt_ph.sh <scene> <root>"
  echo "Example: sh data/tt/prep_tt_ph.sh truck data/tt"
  exit 1
fi

DATA_ROOT=$ROOT/datasets/$SCENE

if [ ! -d "$DATA_ROOT/sparse/0" ]; then
  echo "Error: sparse/0 directory not found at $DATA_ROOT/sparse/0"
  exit 1
fi

# Step 1: Convert binary COLMAP model to text format
mkdir -p "$DATA_ROOT/colmap"
colmap model_converter \
    --input_path "$DATA_ROOT/sparse/0" \
    --output_path "$DATA_ROOT/colmap" \
    --output_type TXT

echo "Converted binary COLMAP model to text format in $DATA_ROOT/colmap"

# Step 2: Copy images to resized_images/ with sequential naming and update images.txt
python "$ROOT/prep_tt_ph.py" --path "$DATA_ROOT"

echo "Done. SCNT-formatted dataset ready at: $DATA_ROOT"
