# Video Colorization with Tracktention

This repo extends [DDColor](https://github.com/piddnad/DDColor) to video by injecting **Tracktention** modules into the DDColor backbone. Tracktention uses point tracks (computed with [CoTracker3](https://github.com/facebookresearch/co-tracker)) to propagate color information across frames, improving temporal consistency without any post-processing.

## Results: Automatic Video Colorization
We achieved the following results using 75% fewer training iterations (4x less) than the original paper.


## Setup

Clone the repo with submodules and install dependencies:

```bash
git clone --recurse-submodules <repo-url>
cd <repo>
pip install -r requirements.txt
```

Download a pretrained DDColor checkpoint and place it under `DDColor/pretrain/`.

## Preprocessing

Before training, extract point tracks from your video frames using CoTracker3. Frames should be organized as:

```
<frames_root>/<split>/JPEGImages/<video_id>/<frame>.jpg
```

Then run:

```bash
python preprocessing/extract_tracks.py \
    --input-root <frames_root>/<split>/JPEGImages \
    --output-root <tracks_root>/<split>/JPEGImages \
    --image-size 256 \
    --num-queries 576
```

This writes a `tracks.pt` file alongside each video folder.

## Training

Edit `configs/train_config.yaml` to point `data.frames_root`, `data.tracks_root`, and `ddcolor.root` at your local paths, then run:

```bash
python training/train.py configs/train_config.yaml
```

To resume from a checkpoint:

```bash
python training/train.py configs/train_config.yaml --resume <output_dir>/last.pt
```

Checkpoints and a copy of the config are saved to `system.output_dir` as defined in the YAML.

## Inference

```bash
python inference.py \
    --input <path/to/video_frames_dir> \
    --checkpoint <output_dir>/last.pt \
    --config configs/train_config.yaml \
    --output <path/to/output_dir>
```
