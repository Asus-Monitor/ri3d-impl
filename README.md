# RI3D Implementation

Implementation of [RI3D](https://people.engr.tamu.edu/nimak/Papers/RI3D/index.html) - state of the art few-shot gaussian splatting.

Makes a gaussian splat from very few images (down to 3). Existing solutions (COLMAP + postshot) look terrible with few cameras - holes, floaters, and they're picky about camera placement. COLMAP needs ~12 cameras minimum for a full body scan and still produces results that need manual cleanup.

RI3D cleans up the floaters and inpaints holes with stable diffusion automatically. 

## Training

Since we are just fine tuning models, you can train it yourself on a consumer level gpu. Works on my RTX 2060 Super. Cuda GPU required.

All commands run from `src/`. Paths default to `<project_root>/dataset` and `<project_root>/output`.

**Preprocessing** (Steps 1-4):

```bash
python src/run_pipeline.py --prep
```

**Training all models** (Steps 5+7):

```bash
python src/run_pipeline.py --train_models
```

### Dataset

```bash
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip -O 360_v2.zip
./extract_dataset.sh
```

Downloads & extracts only the images from each scene in `360_v2` into `dataset/`. Requires `p7zip` (`7z` command).

## Usage

All commands run from `src/`.

**Steps 6+8** (make splat scene):

```bash
python src/run_pipeline.py --optimize --scene ../dataset/garden --n_views "A,B,C"
```
