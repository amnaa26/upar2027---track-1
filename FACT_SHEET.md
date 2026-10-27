# Fact Sheet - UPAR 2027 Challenge, Track 1

**Team / participant name (Codabench):** urahara_senpai
**Members, affiliation, contact:** [YOUR FULL NAME], [YOUR UNIVERSITY], k230066@nu.edu.pk
**Final submission:** ID 1010126, `my_submission_clip.zip`
**Leaderboard result:** Challenge_Avg 0.7712 (mA 0.7736, Label_F1 0.5695, Inst_Acc 0.6511, Inst_Prec 0.7301, Inst_Rec 0.8280, Inst_F1 0.7688)

## 1. Method summary
A single CLIP ViT-B/16 image encoder (pretrained OpenAI weights) fine-tuned end to end for the 40 binary attributes. A linear head on the CLS feature gives 40 logits; sigmoid gives probabilities. Input size is 256x128 (the positional embedding was resized bicubically to a 16x8 patch grid). At test time the logits of the image and its mirror image are averaged, and an attribute is predicted present when its probability is above 0.45.

## 2. Training details
- Loss: binary cross-entropy with a mask so unknown labels (attributes not annotated in a source dataset) are ignored; positive-class weight = (negatives/positives)^0.75 (capped) to help rare attributes.
- Augmentation: random crop/flip plus strong degradation to imitate unseen cameras (low resolution, blur, JPEG compression, brightness changes; no hue change).
- Optimizer: AdamW (betas 0.9/0.98), layer-wise learning-rate decay 0.75, base learning rate 4e-5, head learning rate 1e-3, fp16 mixed precision, 4 epochs.
- Final model trained on train + validation images (validation was used only for the final model, after the settings were chosen).
- Settings were chosen with held-out-collection experiments (train on two collections, test on the third) to simulate the domain shift of the hidden test set.

## 3. Data
Only the organizer-provided UPAR training and validation data. No test images, no additional datasets. External pretrained weights: OpenAI CLIP ViT-B/16 (allowed foundation model).

## 4. Compute
Kaggle, NVIDIA T4 GPU. Training of the final model took roughly 1.5 hours. Inference is fast and fits well within the 5-minute limit on the A10 (checked on 256 photos).

## 5. Other models tried
ConvNeXt-Tiny (score 0.7316 on the leaderboard), ConvNeXt-Small, and an ensemble of ConvNeXt-Small + CLIP (not submitted).

## 6. Code
See `README.md`, `train.py`, `clipvit.py`, `run.py`, `upar_final_kaggle.ipynb`.
