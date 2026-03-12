# Train and Eval Documentation for 3DGEER
### Train Examples
> Will be soon released.
### Render and Eval Examples
ScanNet++ Dataset
```bash
bash scripts/render_scnt.sh 1d003b07bd/dslr data/scnt/datasets ckpt/scnt KB
bash scripts/eval_scnt.sh 1d003b07bd/dslr data/scnt/datasets ckpt/scnt KB
bash scripts/eval_scnt.sh 1d003b07bd/dslr data/scnt/datasets ckpt/scnt BEAP
```
Aria Dataset
```bash
bash scripts/render_scnt.sh steakhouse data/aria/scannetpp_formatted ckpt/aria KB
bash scripts/eval_scnt.sh steakhouse data/aria/scannetpp_formatted ckpt/aria KB
bash scripts/eval_scnt.sh steakhouse data/aria/scannetpp_formatted ckpt/aria BEAP
```
Tanks and Temples
```bash
bash scripts/render_scnt.sh truck data/tt/datasets ckpt/tt PH
bash scripts/eval_scnt.sh truck data/tt/datasets ckpt/tt PH
bash scripts/eval_scnt.sh truck data/tt/datasets ckpt/tt BEAP
```
### Evaluation Protocol

Unlike Fisheye-GS and 3DGUT, which remap the ScanNet++ ground-truth images into an equidistant projection space for evaluation, we perform evaluation directly under the original camera model (KB) of the dataset.

For methods using different projection models (e.g., equidistant, pinhole, or BEAP), the rendered outputs must be remapped back to the original camera space before comparison with the ground truth.

However, such remapping inevitably introduces pixel-level interpolation shifts, which can distort perceptual metrics (e.g., PSNR / SSIM / LPIPS) and prevent them from faithfully reflecting the actual scene reconstruction quality.

To ensure pixel-wise alignment, we apply the same remapping operation to the ground-truth images as well. This guarantees that both predictions and ground truth undergo identical interpolation, eliminating evaluation bias caused by projection conversion.

For the ray mapping used in this process (to reproduce the numbers in Table K.2), we use the grid map provided by:

https://huggingface.co/yuliangguo/depth-any-camera/blob/main/scannetpp_dac_swinl_indoor_2025_01.zip

Qualitative comparisons can be found in Figure K.4.

<p align="center">
  <img src="../assets/scnt_comp.jpg" alt="teaser" style="width: 100%;">
  Figure K.4.; https://arxiv.org/abs/2505.24053
</p>