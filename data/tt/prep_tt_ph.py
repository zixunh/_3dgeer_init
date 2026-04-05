# Convert Tanks and Temples (pinhole) dataset into the ScanNet++-formatted directory structure.
#
# Input layout (raw TT download):
#   <scene>/
#     images/
#     sparse/0/
#       cameras.bin
#       images.bin
#       points3D.bin
#
# Output layout (SCNT-compatible):
#   <scene>/
#     colmap/
#       cameras.txt
#       cameras_fish.txt   (copy of cameras.txt; TT uses perspective cameras)
#       images.txt         (image names updated to sequential 000000.jpg …)
#       points3D.txt
#     resized_images/
#       000000.jpg
#       000001.jpg
#       …

import shutil
from pathlib import Path
from argparse import ArgumentParser


def read_images_txt(path):
    """Return (header_lines, image_tuples) from a COLMAP images.txt.

    Each image tuple: (image_id, line1_str, line2_str, image_name)
    """
    header = []
    images = []
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            header.append(line)
            i += 1
            continue
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        parts = stripped.split()
        image_id = int(parts[0])
        image_name = parts[-1]
        line1 = stripped
        line2 = lines[i + 1].strip() if i + 1 < len(lines) else ""
        images.append((image_id, line1, line2, image_name))
        i += 2

    return header, images


def write_images_txt(path, header, images):
    """Write a COLMAP images.txt with updated image tuples."""
    with open(path, "w") as f:
        f.writelines(header)
        for _, line1, line2, _ in images:
            f.write(line1 + "\n")
            f.write(line2 + "\n")


def main(args):
    root_dir = Path(args.path)
    src_image_dir = root_dir / "images"
    dst_image_dir = root_dir / "resized_images"
    colmap_dir = root_dir / "colmap"
    images_txt_path = colmap_dir / "images.txt"
    cameras_txt_path = colmap_dir / "cameras.txt"
    cameras_fish_txt_path = colmap_dir / "cameras_fish.txt"

    if not images_txt_path.exists():
        raise FileNotFoundError(
            f"images.txt not found at {images_txt_path}. "
            "Run colmap model_converter first (or use prep_tt_ph.sh)."
        )

    dst_image_dir.mkdir(parents=True, exist_ok=True)

    # Read and sort images by COLMAP image_id
    header, images = read_images_txt(images_txt_path)
    images.sort(key=lambda x: x[0])

    updated_images = []
    for idx, (image_id, line1, line2, old_name) in enumerate(images):
        ext = Path(old_name).suffix or ".jpg"
        new_name = f"{idx:06d}{ext}"

        src_path = src_image_dir / old_name
        if not src_path.exists():
            # Fallback: search by basename only (handles subdirectory prefixes)
            src_path = src_image_dir / Path(old_name).name
        if src_path.exists():
            shutil.copy2(str(src_path), str(dst_image_dir / new_name))
        else:
            print(f"Warning: source image not found: {src_image_dir / old_name}")

        # Update the NAME field in line1
        parts = line1.split()
        parts[-1] = new_name
        updated_images.append((image_id, " ".join(parts), line2, new_name))

    write_images_txt(images_txt_path, header, updated_images)

    # cameras_fish.txt — TT uses perspective (pinhole/radial) cameras,
    # so no fisheye conversion is needed; just provide an identical copy.
    if cameras_txt_path.exists() and not cameras_fish_txt_path.exists():
        shutil.copy2(str(cameras_txt_path), str(cameras_fish_txt_path))
        print(f"Created cameras_fish.txt from cameras.txt")

    print(f"Processed {len(images)} images.")
    print(f"Images saved to:  {dst_image_dir}")
    print(f"Updated COLMAP:   {images_txt_path}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert a Tanks-and-Temples scene into ScanNet++ directory format."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the scene directory, e.g. data/tt/datasets/truck",
    )
    args = parser.parse_args()
    main(args)
