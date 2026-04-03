# Extensive Experimental Report: CGSIL and CGSIL v2 on Long-Tailed CIFAR-10/100

## Table of Contents

1. [Introduction](#introduction)
2. [Experimental Scope](#experimental-scope)
3. [Methodology](#methodology)
   1. [Datasets and Long-Tailed Construction](#datasets-and-long-tailed-construction)
   2. [Backbone and Training Protocol](#backbone-and-training-protocol)
   3. [Evaluation Metrics and Model Selection](#evaluation-metrics-and-model-selection)
   4. [CGSIL Formulation in This Repository](#cgsil-formulation-in-this-repository)
   5. [Definition of Every CGSIL Variant](#definition-of-every-cgsil-variant)
4. [Results](#results)
   1. [Overall Winners](#overall-winners)
   2. [Core Method Comparison](#core-method-comparison)
   3. [CGSIL v2 Ablation Study](#cgsil-v2-ablation-study)
   4. [Quantitative Takeaways](#quantitative-takeaways)
5. [Discussion](#discussion)
   1. [What Improved from CGSIL v1 to CGSIL v2](#what-improved-from-cgsil-v1-to-cgsil-v2)
   2. [Why the Best Configuration Depends on the Dataset](#why-the-best-configuration-depends-on-the-dataset)
   3. [What the Ablations Reveal](#what-the-ablations-reveal)
   4. [Failure Modes and Caveats](#failure-modes-and-caveats)
   5. [Recommended Next Experiments](#recommended-next-experiments)
6. [Conclusion](#conclusion)

## Introduction

Class imbalance remains one of the central practical challenges in supervised visual recognition. On long-tailed datasets, standard empirical risk minimization often overfits to high-frequency head classes, while low-frequency tail classes suffer from both representation under-learning and classifier bias. In this setting, re-weighting and focal-style losses are often effective because they change the optimization signal in favor of minority classes. However, these methods do not directly address a second problem: **head and tail classes can induce conflicting gradients during training**.

The motivation behind CGSIL is to explicitly intervene at the gradient level. Rather than only changing the loss weights, CGSIL estimates gradients associated with head and tail groups and applies a projection step intended to remove destructive interference. Conceptually, if the head-group gradient opposes the tail-group gradient, the head component can be projected to reduce negative transfer. This gives the tail signal more room to shape the update.

In the initial implementation, however, CGSIL underperformed strong baselines such as weighted cross-entropy and focal loss, especially under moderate imbalance. This led to a second phase of experimentation focused on refining the method rather than discarding it. The resulting **CGSIL v2** introduced a set of practical controls designed to stabilize and focus the surgery mechanism:

- **Warmup before surgery**
- **Activation gating based on tail support in the current batch**
- **Conflict thresholding before projection**
- **Configurable base loss under CGSIL**
- **Restriction of surgery to either the full network or only the classifier head**

This report documents that refinement process end to end. It explains the implemented training pipeline, defines every CGSIL variant evaluated, summarizes the final experimental results, and discusses what the ablation study implies about when gradient surgery helps and how it should be configured.

## Experimental Scope

The repository contains **70 non-smoke experiment outputs** used for this analysis.

- **48 runs**
  - The full CGSIL v2 matrix:
  - 2 datasets (`cifar10`, `cifar100`)
  - 3 imbalance ratios (`10`, `50`, `100`)
  - 8 CGSIL v2 configurations per dataset/ratio

- **18 runs**
  - Baseline methods:
  - `erm`, `weighted_ce`, and `focal`
  - Across both datasets and all three imbalance ratios

- **4 runs**
  - Historical CGSIL v1 outputs available from earlier experimentation

Of these, **69 runs completed the full 200-epoch schedule**. One historical run, `cifar100_cgsil_ir10_seed42`, logged **173 epochs** rather than 200. Its best recorded score is still reported, but comparisons involving that specific historical run should be interpreted cautiously.

All experiments used a **ResNet-18** backbone.

## Methodology

### Datasets and Long-Tailed Construction

This repository supports `cifar10` and `cifar100` through `torchvision`. Long-tailed training sets are generated synthetically from the standard training split.

- **Training dataset source**
  - `torchvision.datasets.CIFAR10` or `CIFAR100`

- **Test dataset source**
  - Standard balanced CIFAR test split

- **Long-tail construction**
  - Implemented in `datasets/cifar_lt.py`
  - Uses class-wise subsampling of the original training set
  - Default imbalance type is `exp`

- **Per-class image counts**
  - Let `img_max = total_images / num_classes`
  - For class index `c`, the exponential profile uses:
  - `img_max * imbalance_ratio^(-c / (num_classes - 1))`

- **Imbalance ratios evaluated**
  - `10`
  - `50`
  - `100`

- **Tail class definition**
  - Tail classes are defined as the bottom `tail_quantile` fraction by training count
  - In these experiments, the default quantile was `0.3`
  - Therefore, the rarest 30% of classes are treated as tail classes for CGSIL grouping

This design means that evaluation is always conducted on the standard balanced CIFAR test set, while the model is trained on a deliberately long-tailed training distribution.

### Backbone and Training Protocol

The training script is `train_cifar_lt.py`.

- **Architecture**
  - `torchvision.models.resnet18(num_classes=num_classes)`
  - Adapted for CIFAR resolution by replacing the ImageNet stem:
    - `conv1` becomes `3x3`, stride `1`, padding `1`
    - `maxpool` is replaced with `Identity`

- **Data augmentation**
  - Training:
    - random crop with padding `4`
    - random horizontal flip
    - normalization with CIFAR mean/std
  - Evaluation:
    - tensor conversion
    - normalization with the same mean/std

- **Optimization**
  - Optimizer: SGD with Nesterov momentum
  - Learning rate: `0.1`
  - Momentum: `0.9`
  - Weight decay: `5e-4`
  - Epochs: `200`
  - Batch size: `128`

- **Learning-rate schedule**
  - Warmup + cosine decay
  - LR warmup epochs: default `5`
  - Minimum LR floor: `1e-5`

- **Randomness control**
  - Seed fixed to `42`
  - Python, NumPy, and PyTorch seeds are set

### Evaluation Metrics and Model Selection

For every epoch, the training script logs:

- **Accuracy**
- **Macro-F1**
- **Balanced accuracy**
- **Macro precision**
- **Per-class recall**
- **Per-class precision**
- **CGSIL-specific diagnostics**
  - gradient dot product
  - cosine similarity
  - beta value
  - fraction of batches where CGSIL was active
  - fraction of batches where projection was applied
  - number of tail/head classes present in the batch
  - tail sample count per batch

A checkpoint is saved whenever **balanced accuracy** improves. Accordingly, all result tables in this report use the **best epoch by balanced accuracy**, rather than the final epoch, because that is the model-selection rule implemented in the training pipeline.

### CGSIL Formulation in This Repository

The gradient surgery implementation lives in `grad_surgery.py`.

The core logic is:

1. Compute a per-sample loss vector.
2. Aggregate per-sample losses into **per-class mean losses** within the current batch.
3. Compute one gradient vector per class via autograd.
4. Partition the present classes in the batch into:
   - **tail classes**
   - **head classes**
5. Average per-class gradients within each group to obtain:
   - `g_tail`
   - `g_head`
6. If both groups are present, check whether `g_head` conflicts with `g_tail`.
7. When conflict is detected, project the head gradient away from the tail gradient.
8. Form the final gradient:
   - `beta * g_tail + (1 - beta) * g_head_projected`

The implemented conflict criterion is:

- Projection is only applied when:
  - the dot product is negative
  - the cosine similarity is below `conflict_threshold`
  - the tail gradient norm is non-negligible

This is more conservative than projecting on every mild disagreement. It lets the training loop treat some small or noisy head-tail disagreements as acceptable.

The coefficient `beta` is linearly scheduled from `beta_start` to `beta_end` over the course of training. Intuitively, higher `beta` gives more relative weight to the tail gradient component.

### Definition of Every CGSIL Variant

This section is the methodological core of the report. The naming used below matches the logged experiment names.

#### 1. `cgsil_v1`

This is the baseline CGSIL implementation from the earlier phase of experimentation, before the v2 controls were added and systematically explored. It serves as the main historical reference point.

Interpretation:

- **Purpose**
  - Baseline gradient-surgery method before refinement

- **Role in this report**
  - Establish how much v2 improved over the original approach

#### 2. `cgsilv2_legacy`

This is the v2 code path configured to behave as closely as possible to the original, simple CGSIL recipe.

- **Base loss**
  - `ce`

- **Surgery scope**
  - `all`

- **Warmup before surgery**
  - `0`

- **Tail support gating**
  - effectively minimal
  - `min_tail_samples = 1`
  - `min_tail_classes = 1`

- **Conflict threshold**
  - `0.0`

- **Beta schedule**
  - `0.8 -> 0.9`

Interpretation:

- **Purpose**
  - Separate improvements due to the new implementation machinery from improvements due to the tuned v2 design choices

#### 3. `cgsilv2_main`

This is the proposed practical v2 configuration.

- **Base loss**
  - `weighted_ce`

- **Surgery scope**
  - `fc`

- **Warmup before surgery**
  - `50` epochs

- **Tail support gating**
  - `min_tail_samples = 8`
  - `min_tail_classes = 2`

- **Conflict threshold**
  - `-0.05`

- **Beta schedule**
  - `0.6 -> 0.8`

Interpretation:

- **Purpose**
  - Use a stronger imbalance-aware base loss
  - delay surgery until basic representation learning stabilizes
  - activate surgery only when enough tail evidence is present
  - restrict surgery to the classifier head where class competition is most direct
  - avoid projecting on weak or noisy conflicts

#### 4. `cgsilv2_ablate_base_ce`

Same as `cgsilv2_main`, except the base loss is changed from `weighted_ce` to plain `ce`.

- **Purpose**
  - Test whether CGSIL should operate on top of an unweighted classification objective

#### 5. `cgsilv2_ablate_base_focal`

Same as `cgsilv2_main`, except the base loss is changed to `focal` with `gamma = 2.0`.

- **Purpose**
  - Test whether CGSIL combines better with a hard-example focusing loss than with weighted cross-entropy

#### 6. `cgsilv2_ablate_scope_all`

Same as `cgsilv2_main`, except the surgery scope is changed from `fc` to `all` parameters.

- **Purpose**
  - Determine whether surgery should shape only the classifier or the entire representation stack

#### 7. `cgsilv2_ablate_nowarmup`

Same as `cgsilv2_main`, except surgery is allowed from the start of training.

- **Purpose**
  - Test whether delaying surgery is important for stability and representation formation

#### 8. `cgsilv2_ablate_loose_tail`

Same as `cgsilv2_main`, except the activation gate is relaxed.

- **Settings**
  - `min_tail_samples = 1`
  - `min_tail_classes = 1`

- **Purpose**
  - Test whether the stricter batch-support gate is genuinely useful or unnecessarily conservative

#### 9. `cgsilv2_ablate_no_threshold`

Same as `cgsilv2_main`, except the conflict threshold is changed from `-0.05` to `0.0`.

- **Purpose**
  - Test whether surgery should be triggered on any negative conflict rather than only stronger conflicts

## Results

### Overall Winners

| Dataset | IR | Winner | Balanced Acc. | Acc. | Macro-F1 | Best Epoch |
|---|---:|---|---:|---:|---:|---:|
| cifar10 | 10 | weighted_ce | 0.9059 | 0.9059 | 0.9063 | 189 |
| cifar10 | 50 | cgsilv2_ablate_base_focal | 0.8175 | 0.8175 | 0.8186 | 145 |
| cifar10 | 100 | cgsilv2_ablate_no_threshold | 0.7687 | 0.7687 | 0.7695 | 142 |
| cifar100 | 10 | cgsilv2_ablate_base_ce | 0.6469 | 0.6469 | 0.6434 | 193 |
| cifar100 | 50 | cgsilv2_ablate_base_ce | 0.4954 | 0.4954 | 0.4679 | 126 |
| cifar100 | 100 | cgsilv2_ablate_base_ce | 0.4401 | 0.4401 | 0.3955 | 153 |

At a high level, the final leaderboard already suggests the main pattern of the study:

- **CIFAR-10 with mild imbalance**
  - classical re-weighting remains strongest

- **CIFAR-10 with stronger imbalance**
  - CGSIL v2 becomes competitive or best-in-class

- **CIFAR-100**
  - CGSIL v2 can outperform baselines, but the best configuration is not the nominal `main` recipe
  - instead, the strongest setting is the `base_ce` ablation

### Core Method Comparison

This table compares the primary baseline methods with the three central CGSIL points of reference: historical `cgsil_v1`, `cgsilv2_legacy`, and `cgsilv2_main`.

### CIFAR-10

| IR | ERM | Weighted CE | Focal | CGSIL v1 | CGSIL v2 legacy | CGSIL v2 main |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.9020 | 0.9059 | 0.9018 | 0.8450 | 0.8421 | 0.8348 |
| 50 | 0.8158 | 0.8026 | 0.8080 | 0.6182 | 0.6303 | 0.8138 |
| 100 | 0.7556 | 0.7486 | 0.7596 | 0.4372 | 0.4994 | 0.7650 |

### CIFAR-100

| IR | ERM | Weighted CE | Focal | CGSIL v1 | CGSIL v2 legacy | CGSIL v2 main |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.6418 | 0.6383 | 0.6397 | 0.5197 | 0.5227 | 0.6340 |
| 50 | 0.4905 | 0.4480 | 0.4589 | - | 0.2331 | 0.4703 |
| 100 | 0.4380 | 0.3503 | 0.3756 | - | 0.1668 | 0.3736 |

#### Immediate observations

- **CIFAR-10, IR=10**
  - `cgsilv2_main` is clearly below all three conventional baselines
  - gradient surgery is not helping enough in the easier imbalance regime

- **CIFAR-10, IR=50**
  - `cgsilv2_main` nearly matches the strongest baseline
  - it is within `0.0020` balanced accuracy of the best baseline

- **CIFAR-10, IR=100**
  - `cgsilv2_main` becomes the best of the core methods
  - it beats the best baseline by `+0.0054`

- **CIFAR-100, IR=10**
  - `cgsilv2_main` almost matches the strongest baseline but does not surpass it

- **CIFAR-100, IR=50` and `IR=100`**
  - `cgsilv2_main` is a substantial improvement over `legacy`
  - but it still trails the strongest baseline and trails the best v2 ablation even more clearly

### CGSIL v2 Ablation Study

### CIFAR-10

| IR | Legacy | Main | Base CE | Base Focal | Scope All | No Warmup | Loose Tail | No Threshold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.8421 | 0.8348 | 0.8119 | 0.8415 | 0.8245 | 0.8214 | 0.8381 | 0.8458 |
| 50 | 0.6303 | 0.8138 | 0.8103 | 0.8175 | 0.7261 | 0.7967 | 0.7509 | 0.8069 |
| 100 | 0.4994 | 0.7650 | 0.7488 | 0.7614 | 0.7002 | 0.7521 | 0.6640 | 0.7687 |

### CIFAR-100

| IR | Legacy | Main | Base CE | Base Focal | Scope All | No Warmup | Loose Tail | No Threshold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.5227 | 0.6340 | 0.6469 | 0.6361 | 0.5095 | 0.6084 | 0.6313 | 0.6343 |
| 50 | 0.2331 | 0.4703 | 0.4954 | 0.4634 | 0.3609 | 0.4486 | 0.3577 | 0.4707 |
| 100 | 0.1668 | 0.3736 | 0.4401 | 0.3812 | 0.2733 | 0.3629 | 0.2674 | 0.3911 |

These ablations are the most informative part of the study, because they explain **why** the nominal v2 configuration sometimes wins and sometimes does not.

### Quantitative Takeaways

#### 1. Improvement from `cgsil_v1` to `cgsilv2_main`

Where historical v1 numbers are available, the gains are substantial except in the easiest CIFAR-10 setting.

| Dataset | IR | Delta (`v2 main - v1`) |
|---|---:|---:|
| cifar10 | 10 | -0.0102 |
| cifar10 | 50 | +0.1956 |
| cifar10 | 100 | +0.3278 |
| cifar100 | 10 | +0.1143 |

#### 2. Improvement from `cgsilv2_legacy` to `cgsilv2_main`

This isolates the effect of the v2 design choices.

| Dataset | IR | Delta (`main - legacy`) |
|---|---:|---:|
| cifar10 | 10 | -0.0073 |
| cifar10 | 50 | +0.1835 |
| cifar10 | 100 | +0.2656 |
| cifar100 | 10 | +0.1113 |
| cifar100 | 50 | +0.2372 |
| cifar100 | 100 | +0.2068 |

This table is one of the strongest pieces of evidence in the report. It shows that the refinements in v2 were not cosmetic: under moderate and severe imbalance, they transformed CGSIL from a weak method into a serious competitor.

#### 3. Gap between `cgsilv2_main` and the best baseline

| Dataset | IR | Delta (`v2 main - best baseline`) |
|---|---:|---:|
| cifar10 | 10 | -0.0711 |
| cifar10 | 50 | -0.0020 |
| cifar10 | 100 | +0.0054 |
| cifar100 | 10 | -0.0078 |
| cifar100 | 50 | -0.0202 |
| cifar100 | 100 | -0.0644 |

Interpretation:

- **CIFAR-10 IR=100**
  - `cgsilv2_main` is genuinely better than the best conventional baseline

- **CIFAR-10 IR=50` and `CIFAR-100 IR=10`**
  - `cgsilv2_main` is close enough to be considered competitive

- **CIFAR-10 IR=10` and `CIFAR-100 IR=100`**
  - it is not yet the best default choice

#### 4. Average ablation ranking by dataset

Averaging balanced accuracy across imbalance ratios provides a rough sense of which design choices are robust.

##### CIFAR-10 average balanced accuracy across IR={10,50,100}

| Variant | Mean BA |
|---|---:|
| cgsilv2_ablate_no_threshold | 0.8071 |
| cgsilv2_ablate_base_focal | 0.8068 |
| cgsilv2_main | 0.8045 |
| cgsilv2_ablate_base_ce | 0.7903 |
| cgsilv2_ablate_nowarmup | 0.7901 |
| cgsilv2_ablate_loose_tail | 0.7510 |
| cgsilv2_ablate_scope_all | 0.7503 |
| cgsilv2_legacy | 0.6573 |

##### CIFAR-100 average balanced accuracy across IR={10,50,100}

| Variant | Mean BA |
|---|---:|
| cgsilv2_ablate_base_ce | 0.5275 |
| cgsilv2_ablate_no_threshold | 0.4987 |
| cgsilv2_ablate_base_focal | 0.4936 |
| cgsilv2_main | 0.4926 |
| cgsilv2_ablate_nowarmup | 0.4733 |
| cgsilv2_ablate_loose_tail | 0.4188 |
| cgsilv2_ablate_scope_all | 0.3812 |
| cgsilv2_legacy | 0.3075 |

## Discussion

### What Improved from CGSIL v1 to CGSIL v2

The original CGSIL idea was reasonable: identify destructive interference between head and tail groups and neutralize part of it via gradient projection. But the early version applied this logic too broadly and too eagerly. Several v2 refinements appear to have corrected that.

- **Warmup helps decouple representation learning from early surgery noise**
  - Early in training, class-wise gradients are unstable and the batch composition is highly variable
  - Delaying surgery allows the backbone and classifier to learn a usable organization before manipulating gradients

- **Support gating prevents tail-driven updates from becoming too noisy**
  - In highly imbalanced mini-batches, tail evidence can be too sparse to estimate a reliable group gradient
  - Requiring multiple tail samples and at least two tail classes acts like a confidence filter

- **Restricting surgery to the classifier head is usually better than global surgery**
  - This is one of the clearest patterns in the results
  - `scope_all` consistently underperforms `fc`
  - This suggests that head-tail conflict is more productively handled at the classifier boundary than by perturbing the entire feature hierarchy

- **The base loss matters a lot**
  - Gradient surgery is not independent of the base objective
  - Different datasets appear to prefer different base losses under the same surgery mechanism

### Why the Best Configuration Depends on the Dataset

One of the most important empirical outcomes is that **there is no single universal CGSIL v2 configuration**.

#### CIFAR-10

CIFAR-10 has relatively few classes and each class retains more effective support even under imbalance than CIFAR-100. In this regime:

- weighted CE is already extremely strong when imbalance is only moderate
- focal and no-threshold surgery become more useful as imbalance becomes more severe
- the main v2 recipe is already good, but the best variants are often:
  - `base_focal`
  - `no_threshold`

This suggests that for CIFAR-10, the optimization problem is not primarily one of insufficient tail emphasis. Instead, it is one of **making the surgery trigger at the right moments** and pairing it with a loss that maintains enough emphasis on informative hard examples.

#### CIFAR-100

CIFAR-100 is much harder. Each class receives fewer examples, the label space is denser, and minority classes are more fragile. In this setting:

- plain CE under surgery outperforms weighted CE under surgery
- relaxing the threshold can help, but not as decisively as switching the base loss to CE
- loose-tail gating and full-network surgery are especially harmful

This indicates that in CIFAR-100, aggressively compounding imbalance-aware weighting with gradient surgery may over-correct the optimization process. Put differently, **the surgery itself already rebalances the effective update enough that an additional weighted base objective may become counterproductive**.

### What the Ablations Reveal

This section interprets each ablation directly.

#### `base_ce`

This was the strongest and most consistent variant on CIFAR-100.

- **Interpretation**
  - In large-class, strongly imbalanced regimes, plain cross-entropy may provide a cleaner base signal for surgery than weighted CE

- **Practical implication**
  - If the dataset is harder and more fragmented, start from `base_ce`

#### `base_focal`

This variant was strongest on CIFAR-10 IR=50 and very competitive elsewhere on CIFAR-10.

- **Interpretation**
  - Focal loss may complement surgery well when imbalance is substantial but the class space is still compact

- **Practical implication**
  - For CIFAR-10-like problems, `focal + fc-scoped surgery` is a strong recipe to keep on the shortlist

#### `scope_all`

This was one of the worst ablations across both datasets.

- **Interpretation**
  - Projecting the full network may interfere too much with shared representation learning
  - not all head-tail disagreement at deep feature layers should be suppressed

- **Practical implication**
  - Restrict surgery to `fc` unless there is a strong reason not to

#### `no_warmup`

Removing warmup usually hurt relative to `main`.

- **Interpretation**
  - Early surgery likely acts on immature gradients and destabilizes training

- **Practical implication**
  - Warmup should be retained as a default component of CGSIL v2

#### `loose_tail`

Relaxing the support gate generally degraded performance, especially in harder settings.

- **Interpretation**
  - Sparse tail participation creates noisy tail-group gradients
  - forcing surgery under such conditions can amplify variance rather than reduce conflict

- **Practical implication**
  - Tail-support gating is not merely a convenience feature; it appears to be structurally important

#### `no_threshold`

This was especially strong on CIFAR-10 and competitive on CIFAR-100.

- **Interpretation**
  - The `-0.05` threshold used in `main` may be too conservative in some regimes
  - projecting on any negative conflict can be beneficial when the head-tail interference signal is strong enough and consistent enough

- **Practical implication**
  - Threshold tuning deserves to be treated as a first-class hyperparameter, not a detail

### Failure Modes and Caveats

No report of this kind is complete without acknowledging what did **not** work or where the interpretation should remain cautious.

- **CGSIL is not uniformly superior**
  - On CIFAR-10 IR=10, standard weighted CE remains decisively better than all CGSIL variants
  - Gradient surgery appears most useful when imbalance is strong enough that direct head-tail competition becomes a dominant training issue

- **One historical run is incomplete**
  - `cifar100_cgsil_ir10_seed42` logged 173 epochs rather than 200
  - Its reported best metric may underestimate what CGSIL v1 could have reached with a full schedule
  - However, the gap to the strongest v2 configurations is large enough that the qualitative conclusion is unlikely to change

- **Single-seed evaluation**
  - All reported numbers are from seed `42`
  - Strong claims about small differences, especially around `0.002` to `0.01` BA, should be validated with multiple seeds

- **No statistical confidence intervals**
  - The ranking is informative, but variance is not yet quantified

- **No runtime/throughput comparison table**
  - This report focuses on predictive performance, not computational overhead
  - CGSIL variants require multiple per-class gradient computations within a batch and therefore carry a nontrivial training cost relative to ERM-like baselines

### Recommended Next Experiments

Based on the current evidence, the following follow-up experiments would be the most informative.

#### 1. Confirmatory multi-seed runs

Repeat the strongest candidates with at least 3 seeds.

- **CIFAR-10 candidates**
  - `cgsilv2_ablate_no_threshold`
  - `cgsilv2_ablate_base_focal`
  - `weighted_ce`

- **CIFAR-100 candidates**
  - `cgsilv2_ablate_base_ce`
  - `erm`
  - `cgsilv2_ablate_no_threshold`

This would tell us whether the observed gains are robust or within seed noise.

#### 2. Combine the best-performing v2 ingredients

The current ablations change one component at a time relative to `main`. The next logical step is to combine the strongest ingredients.

Suggested combinations:

- **For CIFAR-10**
  - `base_focal + no_threshold + fc`

- **For CIFAR-100**
  - `base_ce + no_threshold + fc`

These combinations were not directly tested in the current matrix and could plausibly outperform every existing configuration.

#### 3. Tune the threshold more finely

Only two thresholds were effectively examined:

- `-0.05`
- `0.0`

A more complete sweep such as `{-0.1, -0.05, -0.02, 0.0}` could help identify whether there is a smoother optimum.

#### 4. Tune tail-activation criteria by dataset

The same tail-support gate was used across datasets for the `main` configuration. CIFAR-100 may benefit from a different gate than CIFAR-10.

Potential sweep:

- `min_tail_samples`: `4, 8, 12`
- `min_tail_classes`: `1, 2, 3`

#### 5. Study surgery frequency diagnostics

The logs already contain:

- `cgsil_active`
- `cgsil_projection_applied`
- `cgsil_dot`
- `cgsil_cosine`

A follow-up analysis could correlate these diagnostics with validation improvements. That would clarify whether the best variants win because:

- surgery is triggered more often
- surgery is triggered less often but more meaningfully
- or the base loss alone already does most of the work

## Conclusion

This study began with a practical problem: the initial CGSIL implementation was underperforming simpler, more established baselines on long-tailed CIFAR benchmarks. The experimental evidence now supports a more nuanced conclusion.

**CGSIL as an idea is viable, but its performance depends strongly on how and where the surgery is applied.**

The v2 refinements fundamentally changed the picture:

- They turned CGSIL from a weak method into a competitive one under stronger imbalance.
- They showed that indiscriminate full-network surgery is usually harmful.
- They established that surgery should often be delayed and gated by batch support.
- They revealed strong dataset dependence in the preferred base loss.

The final practical recommendations are straightforward.

- **For CIFAR-10-like settings**
  - The best CGSIL v2 candidates are `no_threshold` and `base_focal`
  - Classical weighted CE remains very strong under milder imbalance

- **For CIFAR-100-like settings**
  - `base_ce` is the most reliable CGSIL v2 configuration
  - Weighted CE is not the best base objective under surgery in this harder regime

In short, the experiments do not support a one-size-fits-all version of CGSIL. Instead, they support **a family of CGSIL v2 configurations whose effectiveness depends on the dataset and the severity of imbalance**. That is a more valuable outcome than merely finding a single winner, because it explains which design choices matter and why.

The repository is now in a much stronger state than at the beginning of the study: it contains a runnable baseline suite, a refined CGSIL implementation, a meaningful ablation framework, and empirical evidence that points clearly toward the next round of improvements.
