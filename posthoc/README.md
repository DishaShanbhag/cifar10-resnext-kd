# Post-training experiments (no change to the training recipe)

Everything here starts from the trained 0.915 student (`notebooks/02_*`) or from the uniform soup of three
fine-tuned copies of it. Nothing here was trained on, tuned on, or adapted to the hidden test images.

**Decision rule used for every row:** a method is adopted only if it beats the reference on the
2,500-image **validation** split by more than 0.2 pt (5 images). The 2,500-image **internal test** split is
only the honest check (noise about ±0.55 pt, ±14 images). The Kaggle public split (~1,000 images) has about
±1 pt of noise and was never used to choose.

References: 0.915 student = val 0.9852 / internal 0.9812. Uniform soup = val 0.9868 / internal 0.9824.

## Results

| Group | Method | val / internal test | Verdict |
|---|---|---|---|
| Weight averaging | **Uniform soup of 3** (0.915 + 100-ep fine-tune + 50-ep tail; Wortsman et al. 2022) | **0.9868 / 0.9824**, public 0.917 | **helped** |
| | WiSE interpolation between soup and 0.915 | 0.9864 / 0.9824 | tie |
| | Big soup of 15 ingredients, uniform | 0.9856 / 0.9812 | tie / worse |
| | Big soup of 15, greedy (cross-fitted) | 0.9864 / 0.9828 | tie |
| | SVD tightening of the soup; class-selective soup | no change | tie |
| BatchNorm | Precise BN on upright train images (0.915 model) | 0.9676–0.9768 | hurt (−1.6 pt) |
| | BN recompute on the soup with training-matched inputs (4 rotations + training augmentation) | best 0.9868 / 0.9820 | tie |
| Test-time augmentation | **Flip TTA on the soup** (vs. no flip) | +7 images val, +4 internal | **helped** |
| | Shifts 1/2 px, 5/10/18 views (0.915 model) | within noise of flip (per-setting numbers not kept) | tie |
| | airbench 6-view TTA (0.915 model) | 0.9848 / 0.9824 | tie |
| | Learned TTA weights; gated TTA; saliency-masked TTA | 0.9848 / 0.9816–0.9824 | tie |
| | Contrast / brightness TTA; anchor-embedding TTA; combinations | 0.9844–0.9848 / 0.9816–0.9820 | tie |
| | Zoom TTA (0.75–1.25), both models | best 0.9856 / 0.9812 | hurt / tie |
| | Blur / sharpen TTA | best 0.9868 / 0.9824 | tie |
| | Bilateral (edge-preserving) filter TTA | best 0.9856 / 0.9824 | tie / hurt |
| | Shift sweep on the soup: **3 px, 4 directions + flips** | 0.9868 / 0.9844, public 0.917 | tie on val (+5 internal); submitted as alternative |
| Calibration | Vector scaling; Dirichlet (ODIR); temperature balancing; class re-weighting | 0.9844–0.9848 / 0.9800 | tie / hurt |
| Feature-space heads | Nearest class mean; cosine head + margin; Mahalanobis (shared / per class); kNN blend; density kNN; PC-Align | 0.976–0.9852 / ≤ 0.9820 | tie / hurt |
| Input preprocessing | ZCA whitening; range normalisation | down to 0.816 / no gain | hurt / tie |
| Fine-tuning | Influence-based cleaning + head retrain | 0.9860 / 0.9788 | hurt on internal |
| | Teacher-ensemble distillation (4 models) | gate failed: ensemble val 0.9864 ≤ 0.9888 | not trained |
| | Teacher-guided last-layer fine-tune | best 0.9868 / 0.9820 | tie |

**Measured side findings (useful for the report):** a Gaussian blur of σ = 0.8 costs about 4 pt on its own,
zoomed views cost 1–3 pt, and a strong bilateral filter costs up to 3 pt. The model leans on fine texture
and on object size, which differ in the CIFAR-10.2-style test set; that would have to be fixed in training
(scale / blur augmentation), not after it.

**Not allowed / not used:** BN statistics or entropy minimisation on the test images (TENT), test-time
training, aligning to test statistics, and any ensemble at inference (the TA ruling requires one network).

## Files

| File | What it does |
|---|---|
| `p01_preciseBN_learnedTTA.ipynb` | Precise BN and learned TTA weights on the 0.915 model (executed) |
| `p02_airbench_tta.ipynb` | airbench-style 6-view TTA (executed) |
| `p03_calibration_vector_dirichlet.ipynb` | vector scaling and Dirichlet-style calibration (executed) |
| `p04_gated_tta_saliency.ipynb` | confidence-gated TTA and saliency-masked second pass (executed) |
| `soup_uniform_and_greedy.py` | builds the uniform and cross-fitted greedy soups, writes a submission |
| `standalone_colour_size_tta.py` | one-cell Colab script: loads data + both models, colour/size TTA |
| `cell_*.py` | notebook cells run after the standalone script (BN recompute, zoom, blur, bilateral, shift sweep, greedy TTA, teacher-ensemble gate, head fine-tune) |

The `.py` files are the cells as they were run in Colab; they expect the Drive layout described in the main README.
