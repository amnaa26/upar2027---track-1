# UPAR 2027 Challenge - Track 1 (Pedestrian Attribute Recognition)

Final submission: `my_submission_clip.zip` = one CLIP ViT-B/16 image encoder fine-tuned with a 40-way sigmoid head.

## Files
- `train.py`   - training script (settings are read from environment variables).
- `clipvit.py` - small CLIP ViT-B/16 implementation (loads OpenAI CLIP weights from open_clip, position embedding resized to a 16x8 patch grid for 256x128 input).
- `run.py`     - Codabench entry point (`load_model`, `predict_batch`); reads `config.json` (model file + cutoff 0.45).
- `upar_final_kaggle.ipynb` - the whole pipeline on Kaggle (T4 x2): download data, check CLIP code against open_clip, train, build zips, smoke-test.

## How to reproduce
1. Run `upar_final_kaggle.ipynb` on Kaggle (GPU T4 x2, Internet on). It clones the official starter kit, downloads the provided data, trains, and builds `my_submission_clip.zip`.
2. The CLIP model used in the submission was trained with:
   `TAG=ft_clip ARCH=clip_vit_b16 AUG=strong POSW=0.75 USE_VAL=1 SANITY_N=4000 EPOCHS=4 python train.py`
3. The zip contains `run.py`, `clipvit.py`, `config.json` (cutoff 0.45), `metadata.yaml` (copied from the official example) and the weights `w0.pt`.

## Data
Only the data provided by the organizers (UPAR train + validation: Market-1501, PA100k, PETA) was used. No test images were used. The only external resource is the pretrained OpenAI CLIP ViT-B/16 image encoder weights (via open_clip).

## Trained weights
Download `my_submission_clip.zip` (contains `w0.pt`, `config.json`) from: [YOUR GOOGLE DRIVE LINK]

To run inference locally, unzip it, then use `run.py` (`load_model()` then `predict_batch(samples)`); `config.json` and `w0.pt` must be in the same folder as `run.py` and `clipvit.py`.

## Requirements
Python 3.10+, torch 2.4.1, torchvision, numpy, Pillow (training also: pandas, tqdm, gdown, requests, pyyaml, open_clip_torch).
