# Data Format Documentation of 3DGEER
## 0. Extra Dependency
```
pip install projectaria-tools==1.6.0
pip install pycolmap==3.10.0
```
## 1. ScanNet++ Dataset
Download the dataset following the official [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/) instructions.
### Expected Directory Structure
```
|_./data/scnt
    |_datasets # e.g., download data into this folder
        |_1d003b07bd
        |   |_colmap
        |   |   |_images.txt
        |   |   |_points3D.txt
        |   |   |_cameras.txt
        |   |   |_...
        |   |_nerfstudio
        |   |   |_transforms.json
        |   |_resized_images
        |       |_000000.jpg
        |       |_000001.jpg
        |       |_...
        |_e3ecd49e2b
        |_...
```

## 2. Aria Dataset
Download Aria scenes from: https://explorer.projectaria.com/aria-scenes

After downloading, preprocess the dataset into ScanNet++-formatted COLMAP structure.
```
|_./data/aria
    |_datasets # e.g., download data into this folder
    |   |_graden
    |   |   |_mps
    |   |   |_recording.vrs
    |   |_steakhouse
    |   |_...
    |_scannetpp_formatted # preprocessed folder
        |_garden
        |   |_colmap
        |   |_nerfstudio
        |   |_resized_images
        |_steakhouse
        |_...
```
### Preprocessing Command
```bash
sh data/aria/prep_aria.sh <SCENE_NAME> <ROOT_PATH>
```
Example:
```bash
sh data/aria/prep_aria.sh garden data/aria
```
This script will:
- Convert VRS + MPS outputs  
- Generate COLMAP model files (`cameras.txt`, `images.txt`, `points3D.txt`)  
- Convert binary model to TXT format  
- Remove `.bin` files  
- Produce ScanNet++-compatible directory structure

## 3. Tanks and Temples
Download Tanks and Temples data from: https://www.tanksandtemples.org/download/

After downloading, each scene should look like:
```
|_./data/tt
    |_datasets
        |_truck
        |   |_images
        |   |_sparse/0
        |       |_cameras.bin
        |       |_images.bin
        |       |_points3D.bin
        |_train
        |_...
```

### Preprocessing Command
```bash
sh data/tt/prep_tt_ph.sh <SCENE_NAME> <ROOT_PATH>
```
Example:
```bash
sh data/tt/prep_tt_ph.sh truck data/tt
```
This script will:
- Convert the binary COLMAP model (`sparse/0/*.bin`) to text format (`colmap/*.txt`)
- Copy scene images from `images/` to `resized_images/` with sequential names (`000000.jpg`, `000001.jpg`, …)
- Update `images.txt` to reflect the new file names
- Create `cameras_fish.txt` (copy of `cameras.txt`; TT uses perspective cameras)

After preprocessing, the scene directory matches the ScanNet++ format:
```
|_./data/tt
    |_datasets
        |_truck
        |   |_colmap
        |   |   |_cameras.txt
        |   |   |_cameras_fish.txt
        |   |   |_images.txt
        |   |   |_points3D.txt
        |   |_resized_images
        |       |_000000.jpg
        |       |_000001.jpg
        |       |_...
        |_train
        |_...
```

