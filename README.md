# ResNeXt-29 under 5M parameters for CIFAR-10, distilled from an SE-WRN teacher

NYU Deep Learning (Fall 2026), Project 1: Kaggle challenge
[cifar-10-res-net-architecture-challenge-dl](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl).
Team **The 5M Club**. Author of this repository: Disha Shanbhag.

| | |
|---|---|
| Final model | **one ResNeXt-29 (4×24d), no attention, 4,998,738 trainable parameters** (limit 5,000,000) |
| Kaggle (2,000 hidden test images) | **public 0.912 · private 0.888** |
| Validation / internal test (2,500 + 2,500 held-out CIFAR-10 train images) | 0.9880 / 0.9828 |
| Training | trained from scratch on the provided 50k train set only; no external data, no pretrained weights |
| Notebook that produced it | [`notebooks/03_final_finetune_100ep.ipynb`](notebooks/03_final_finetune_100ep.ipynb) |
| Parameter count + plots without a GPU | [`notebooks/00_final_model_summary.ipynb`](notebooks/00_final_model_summary.ipynb) (uses `torchsummary`) |

## Pipeline

```
SE-WRN-28-10 + SLA teacher (36.8M params, training only)      notebooks/01   val 0.988 | internal 0.984
        │  knowledge distillation (α = 0.9, T = 1, teacher temperature-calibrated on val)
        ▼
ResNeXt-29 4x24d student, 400 epochs (4,998,738 params)        notebooks/02   val 0.9852 | internal 0.9812 | public 0.915
        │  same recipe, 100 more epochs at peak LR 0.02
        ▼
FINAL: ResNeXt-29 4x24d, 100-epoch fine-tune                   notebooks/03   val 0.9880 | internal 0.9828 | public 0.912 | private 0.888
```

Validation and internal-test accuracies use horizontal-flip TTA on the 10-class head, exactly as submitted.

## Architecture (final model)

ResNeXt-29 for CIFAR (Xie et al. 2017), with stage widths widened to fill the 5M budget:

| Part | Details | Params |
|---|---|---|
| Stem | 3×3 conv, 64 channels, BN, ReLU | 1,856 |
| Stage 1 (32×32) | 3 bottleneck blocks, output 280, grouped 3×3 conv (cardinality 4, inner width 96) | |
| Stage 2 (16×16) | 3 blocks, output 528, inner width 192, stride 2 in the first block | |
| Stage 3 (8×8) | 3 blocks, output 1040, inner width 384, stride 2 in the first block | |
| Skip connections | identity, or 1×1 conv + BN when shape changes | |
| Head | global average pool → linear 1040 → 10 | 10,410 |
| SLA head (training only) | linear 1040 → 40 (class × rotation) | 41,640 |
| **Total trainable** | | **4,998,738** |

The full layer-by-layer table is printed by `torchsummary` in notebook 00. Code: [`src/models.py`](src/models.py).
There is no squeeze-excitation or any other attention in the final network; SE blocks appear only in the
training-only teacher, which the course staff confirmed is allowed for teachers trained from scratch.

**Design choices (pros / cons).**
- *ResNeXt bottlenecks vs. plain ResNet blocks.* Grouped 3×3 convolutions buy more width per parameter.
  At about 5M parameters the ResNeXt-29 and a ResNet [4,4,3] were within 0.2 pt of each other without
  distillation (0.9776 vs 0.9784 val), but the ResNeXt took distillation better: 0.9856 against 0.9816
  for a plain WRN-28 student under the same teacher. Cost: about 61 s per epoch on an A100, slower than
  the plain ResNet.
- *Wide final stage (1040).* Most of the budget sits at 8×8 resolution, which is cheap in compute and helps accuracy.
  We also tried a higher-resolution variant (stride 1 in stage 2) on HPC; it scored 0.906 public, not better.
- *SLA rotation head* (Lee et al. 2020). The extra 40-way head costs 41,640 params at training time and
  adds a self-distillation target. On its own it did not raise val (0.9764 after 200 epochs, against 0.9776 for the same network without it after 400),
  but every distilled student used it, and it is part of the recipe that reached 0.985.

## Training recipe

| | Teacher (notebook 01) | Student (notebook 02) | Final fine-tune (notebook 03) |
|---|---|---|---|
| Network | SE-WRN-28-10 + SLA | ResNeXt-29 4×24d + SLA | same, initialised from notebook 02 |
| Epochs | 250 | 400 | 100 |
| Optimiser | SGD Nesterov, momentum 0.9, wd 5e-4, batch 128 | SGD Nesterov + **ASAM** (ρ 0.5), wd 5e-4, batch 256 | same as student |
| LR | 0.1, 2-epoch warm-up, cosine | 0.2, 5-epoch warm-up, cosine to 0 | 0.02, 1-epoch warm-up, cosine to 0 |
| Augmentation | TrivialAugment, crop + flip, cutout 16, CutMix on 50% of batches | TrivialAugment, crop + flip, cutout 16, 10% grayscale, CutMix 50% | same |
| Loss | SLA joint CE + self-distillation | SLA joint CE (label smoothing 0.1) + 0.9 · KD(teacher) + 0.1 · self-distillation | same |
| Weights kept | best val epoch | EMA (decay 0.999), best val epoch | EMA, best val epoch (77) |
| Time (A100) | ~3.4 h (49 s/epoch) | ~6.8 h (61 s/epoch) | ~1.7 h |

Data split (seed 42, stratified): 45,000 train / 2,500 validation / 2,500 internal test, all from the
provided 50k train set. The validation split chose checkpoints and settings; the internal test split was
only used as an honest check. Notebook 00 and `results/*/results.json` hold the per-class numbers.

## Results

**Main path and submissions**

| Model | Params | Val | Internal test | Kaggle public | Kaggle private |
|---|---|---|---|---|---|
| SE-WRN-28-10 + SLA teacher (not submittable) | 36,778,394 | 0.9880 | 0.9840 | – | – |
| ResNeXt-29 student, KD 400 ep | 4,998,738 | 0.9852 | 0.9812 | 0.915 | ≤ 0.888 |
| **ResNeXt-29 + 100-ep fine-tune (final)** | **4,998,738** | **0.9880** | **0.9828** | **0.912** | **0.888** |
| Uniform weight soup of 3 fine-tunes, flip TTA | 4,998,738 | 0.9868 | 0.9824 | 0.917 | ≤ 0.888 |
| Same soup, 3-px shift + flip TTA | 4,998,738 | 0.9868 | 0.9844 | 0.917 | ≤ 0.888 |
| Same recipe on train + val (47.5k) | 4,998,738 | (val in training) | 0.9832 | 0.906 | – |
| Plain WRN-28 student (59/119/236), same teacher | 4,999,722 | 0.9828 | 0.9828 | 0.897 | – |

**Ablation path** (best validation accuracy of the EMA weights during training, no TTA; from `results/all_runs/*/history.csv`)

| Step | Notebook | Params | Val |
|---|---|---|---|
| ResNet [3,3,3] × [64,128,256], no augmentation, 60 ep | e01 | 4,327,754 | 0.8964 |
| + random crop + flip, 200 ep | e01 | 4,327,754 | 0.9568 |
| + TrivialAugment, 300 ep | e01 | 4,327,754 | 0.9672 |
| + EMA + CutMix, 400 ep | e01 | 4,327,754 | 0.9756 |
| ResNet [4,4,3] + SAM | e02 | 4,697,162 | 0.9784 |
| ResNeXt-29 4×24d + same tricks | e03 | 4,845,002 | 0.9768 |
| ResNeXt-29 widened to 5M, ASAM, cosine, 400 ep | e06 | 4,999,850 | 0.9776 |
| + SLA head (200 ep) | e07 | 4,998,738 | 0.9764 |
| KD from an ensemble of our own models (run `kd_student_..._v2`) | e13 | 4,998,738 | 0.9804 |
| Generation-2 KD (students as teachers) | e14 | 4,998,738 | 0.9788 |
| **KD from the SE-WRN-28-10 + SLA teacher** | 02 | 4,998,738 | **0.9856** |
| + 100-epoch fine-tune (final) | 03 | 4,998,738 | 0.9856 (0.9880 with flip TTA) |

Things that did not help and were stopped or dropped: AugMix + JSD (e08, 0.9600 at epoch 165), a
confusion-penalty loss (e04, 0.9752), Shake-Shake and SupCon teachers (e10, e11), a cat/dog specialist
(e12), a second training seed (e17, stopped at epoch 181). About 45 post-training methods are listed with
their measurements in [`posthoc/README.md`](posthoc/README.md); only weight averaging and flip TTA helped.

## What we learned

1. **The test set is shifted.** Every model loses 7–10 pt between our CIFAR-10 val split and Kaggle.
   This matches Lu et al. (2020) for CIFAR-10.2, where 33 models fall on one line
   (CIFAR-10.2 ≈ 1.34 × CIFAR-10 accuracy + const) and only a better in-distribution model moves you up.
2. **The biggest single gain came from a strong teacher trained from scratch.** Distilling from an
   SE-WRN-28-10 + SLA teacher moved the same student from 0.9804 to 0.9856 val; changing the student
   architecture, seeds or loss tricks did not come close.
3. **Public leaderboard differences under 1 pt were noise.** Each half has about 1,000 images (±1 pt).
   The soup scored 0.917 public against 0.912 for the fine-tune, yet the fine-tune was better on private
   (0.888) and was also the best model on validation.
4. **Post-processing could not fix the shift.** Zoom, blur, colour and filtering views each cost 1–4 pt
   on their own; the model relies on texture and object size, which would need training-time augmentation.

## Reproduce

1. Put the Kaggle data in Google Drive at `MyDrive/cifar10_resnet_challenge/data/` (the notebooks build a decoded cache and the split file on first run).
2. Run on a Colab A100, in order: `notebooks/01_teacher_sewrn28_10_sla.ipynb` → `02_kd_student_resnext29_400ep.ipynb` → `03_final_finetune_100ep.ipynb`.
   Each notebook resumes automatically from Drive if the session disconnects and writes
   `runs/<RUN_NAME>/{best.pt, history.csv, results.json, curves.png, submission_*.csv}`.
3. Notebook 03 section 12 prints final validation and internal-test accuracy and the parameter count; section 13 writes `submission.csv`.

Model weights (`best.pt` for the teacher, student and final model) are on Google Drive, not in this repo:
[Drive folder](https://drive.google.com/drive/folders/1Sc31CrEF35GP3KtKxcHIdkYxEez3hhb9?usp=drive_link),
under `runs/<RUN_NAME>/best.pt`. The final model is
`runs/kd_sewrn_only_student_resnext29_slasd_400ep_v1_ft100_lr0.02/best.pt`.

```
notebooks/   00 summary (params, accuracy, plots) · 01 teacher · 02 student · 03 final fine-tune   (executed, with outputs)
experiments/ e01–e19: every earlier run in the order we tried them (executed; a few were stopped early)
posthoc/     post-training experiments and their results table
results/     config, history, results, curves, confusion matrix and submission CSV of each main run; all_runs/ for the rest
src/         model definitions
```

## References

Methods
- S. Xie et al., *Aggregated Residual Transformations for Deep Neural Networks* (ResNeXt), CVPR 2017.
- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- S. Zagoruyko, N. Komodakis, *Wide Residual Networks*, BMVC 2016.
- J. Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018 (teacher only).
- H. Lee, S. J. Hwang, J. Shin, *Self-supervised Label Augmentation via Input Transformations* (SLA), ICML 2020.
- G. Hinton, O. Vinyals, J. Dean, *Distilling the Knowledge in a Neural Network*, 2015.
- C. Guo et al., *On Calibration of Modern Neural Networks* (temperature scaling), ICML 2017.
- R. Caruana et al., *Ensemble Selection from Libraries of Models*, ICML 2004.
- S. G. Müller, F. Hutter, *TrivialAugment*, ICCV 2021.
- S. Yun et al., *CutMix*, ICCV 2019. T. DeVries, G. Taylor, *Cutout*, 2017.
- C. Szegedy et al., *Rethinking the Inception Architecture* (label smoothing), CVPR 2016.
- P. Foret et al., *Sharpness-Aware Minimization*, ICLR 2021. J. Kwon et al., *ASAM*, ICML 2021.
- I. Loshchilov, F. Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017.
- M. Wortsman et al., *Model Soups*, ICML 2022.
- K. Jordan, *airbench: 94% on CIFAR-10 in 3.29 seconds*, 2024 (TTA scheme in posthoc/p02).

Distribution shift
- S. Lu et al., *Harder or Different? A Closer Look at Distribution Shift in Dataset Reproduction*, ICML 2020 UDL workshop (CIFAR-10.2).
- B. Recht et al., *Do ImageNet Classifiers Generalize to ImageNet?* / *Do CIFAR-10 Classifiers Generalize to CIFAR-10?*, 2018–2019.
- J. Miller et al., *Accuracy on the Line*, ICML 2021.

Tried, not in the final model: AugMix (Hendrycks et al. 2020), Shake-Shake (Gastaldi 2017),
SupCon (Khosla et al. 2020), ResizeMix (Qin et al. 2020), Dirichlet calibration (Kull et al. 2019),
SWA (Izmailov et al. 2018), REPAIR (Jordan et al. 2022), RegMixup (Pinto et al. 2022).

Code: all code in this repository was written for this project, with help from Claude (Anthropic), which
the course allowed. Library implementations used: PyTorch, torchvision (`TrivialAugmentWide`),
scikit-learn (stratified split), `torchsummary`.
