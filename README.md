# Enhancing Vedic Mathematics Education: A Hybrid Approach Using Random Forest for Question Classification and Neural Network-Based Knowledge Tracing in the Presence of Noisy Skill Tags

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange.svg)]()

## Abstract

This repository presents a hybrid computational framework for Vedic mathematics education. The system combines (i) a rule-based engine that implements classical sutras—*Urdhva Tiryagbhyam*, *Nikhilam Navatashcaramam Dashatah*, and *Yavadunam Tavadunikritya Varganca Yojayet*—with (ii) a machine-learning pipeline that classifies arithmetic items by optimal sutra and traces learner mastery under noisy skill labels.

A synthetic corpus of 10,000 operand pairs is generated from explicit proximity-to-base rules. Approximately 20% of skill tags are deliberately corrupted to emulate annotation error. A Random Forest classifier recovers clean skill assignments from engineered numerical features. Cleaned tags are then consumed by a Long Short-Term Memory (LSTM) knowledge-tracing model. Response-level class imbalance is addressed with Focal Loss. Empirically, the hybrid tracer attains an AUC of 0.9822 and recovers minority-class (incorrect-response) F1 of 0.91, approaching the oracle tracer trained on uncorrupted labels (AUC 0.9706).

**Keywords:** Vedic mathematics, knowledge tracing, noisy labels, Random Forest, LSTM, Focal Loss, intelligent tutoring systems

---

## 1. Introduction

Vedic mathematics offers compact mental-arithmetic procedures whose applicability depends on structural properties of the operands (proximity to a power of ten, digit length, signed deviation from a chosen base). In an intelligent tutoring setting two problems arise simultaneously:

1. **Item tagging.** The “correct” sutra for a problem is a latent pedagogical label. Human or heuristic tags are frequently inconsistent.
2. **Mastery estimation.** Knowledge-tracing models assume that each interaction is annotated with a reliable skill identifier. Noise in those identifiers degrades sequence models.

This work treats sutra assignment as a supervised classification problem and knowledge tracing as a sequential prediction problem conditioned on *cleaned* tags. The contribution is therefore architectural: a two-module hybrid in which a tabular classifier regularizes the skill channel of a neural tracer.

---

## 2. Vedic Sutras Implemented

| Sutra | Classical statement (abridged) | Computational role |
|---|---|---|
| *Urdhva Tiryagbhyam* | “Vertically and crosswise” | General multiplication via cross-products and carry propagation |
| *Nikhilam* | “All from 9 and the last from 10” | Multiplication of numbers near a power-of-ten base |
| *Yavadunam* | “Whatever the deficiency, lessen it still further and also set up the square” | Squaring numbers near a power-of-ten base |

The rule-based module (`Implement_a_rule_based_AI_for_Vedic_mathematics.ipynb`) exposes Python functions for each sutra and an interactive console that selects a method and accepts integer operands. Edge cases (zeros, mixed bases, carry overflow on the right-hand product) are handled explicitly.

**Assignment rule used as ground truth.** Both operands are labelled *Nikhilam* if each lies within a fixed $\delta$ of some $10^k$ ($k \in \{1,\ldots,5\}$); otherwise the pair is labelled *Urdhva-Tiryagbhyam*. Squaring tasks use the refined *Yavadunam* base-selection heuristic (nearest relevant power of ten).

---

## 3. Methodology

The research notebook (`Enhancing_Vedic_Mathematics_Education_....ipynb`) is organised in three phases.

### 3.1 Phase 1 — Domain-knowledge corpus

- Draw 10,000 pairs $(a,b)$ with $a,b \sim \mathrm{Unif}\{1,\ldots,1000\}$.
- Assign an *optimal* sutra by the proximity rule above.
- Flip 20% of labels uniformly at random to obtain a *noisy* tag.
- Observed class prior before noise: Nikhilam $\approx 0.31\%$, Urdhva-Tiryagbhyam $\approx 99.69\%$. After noise the apparent Nikhilam rate rises to $\approx 20\%$.

The extreme imbalance is pedagogically faithful: most random multiplications are *not* near a convenient base.

### 3.2 Phase 2 — Random Forest skill classifier

**Features**

- Decimal magnitudes: $\lvert a \rvert_{10}$, $\lvert b \rvert_{10}$
- Absolute and relative proximity to the nearest power of ten
- Magnitude difference and operand-relative difference
- Cross-product / carry-potential descriptors

A `RandomForestClassifier` is trained on the engineered table with the *optimal* tag as the supervised target (or, in robustness checks, the noisy tag). The fitted model is subsequently used as a *tag cleaner*: given only $(a,b)$, it emits a denoised skill identifier $q_t$ for every tutoring interaction.

### 3.3 Phase 3 — Neural knowledge tracing

**Simulation.** Fifty synthetic learners are initialised with independent mastery parameters $\theta_s \in [0.2,0.8]$ per skill. Each learner answers 200 items. Correctness $r_t$ is sampled from a logistic function of current mastery; mastery is updated after every trial.

**Sequence construction.** For each learner one obtains a sequence $\{(q_t,r_t)\}_{t=1}^{T}$. Three tag sources are compared:

| Condition | Source of $q_t$ |
|---|---|
| Cleaned (oracle) | Ground-truth optimal sutra |
| Noisy | 20%-corrupted tag |
| Hybrid | Random Forest prediction |

Sequences are one-hot encoded, concatenated with $r_t$, and padded to length 199.

**Architecture.** Keras `Sequential`:

```
Input → LSTM(64, return_sequences=True) → TimeDistributed(Dense(1, sigmoid))
```

**Imbalance remedies.** Incorrect responses constitute a minority class ($\approx 27\%$). Sample weighting proved insufficient (minority F1 remained 0). Focal Loss

$$
\mathrm{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)
$$

with $\gamma=2.0$, $\alpha=0.25$ restored minority-class learning.

---

## 4. Principal Results

| Model | Training tags | Test setting | AUC | Notes |
|---|---|---|---|---|
| Oracle NNKT | optimal | cleaned | **0.9706** | Upper bound |
| Noisy NNKT | noisy | noisy | 0.9681 | Fails on Nikhilam / incorrect class |
| Hybrid NNKT | RF-cleaned | hybrid | 0.9688 | Recovers oracle-level AUC; still fails on incorrect responses |
| Hybrid + sample weights | RF-cleaned | hybrid | 0.9639 | Minority F1 still 0.00 |
| Hybrid + Focal Loss | RF-cleaned | hybrid | **0.9822** | Acc. 0.9603; incorrect-class P/R/F1 = 0.88 / 0.95 / 0.91 |

**Interpretation.** Random Forest cleaning removes *item-level* label noise almost completely: hybrid AUC matches the oracle. *Response-level* imbalance is a separate failure mode and is solved by Focal Loss rather than by reweighting.

---

## 5. Repository Contents

```
.
├── Implement_a_rule_based_AI_for_Vedic_mathematics.ipynb
│     Rule-based sutras + interactive CLI
├── Enhancing_Vedic_Mathematics_Education_A_Hybrid_Approach_....ipynb
│     Data generation, RF classifier, student simulation, NNKT experiments
└── README.md
```

Expected runtime artefacts (generated inside the research notebook, not committed by default):

- `df_operands` — 10k-row feature table with optimal and noisy labels
- Fitted `RandomForestClassifier`
- Padded tensors `X_*`, `y_*` for cleaned / noisy / hybrid splits
- ROC plots and classification reports for each tracer variant

---

## 6. Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn jupyter
```

TensorFlow 2.x is required for the LSTM tracer. CPU execution is sufficient for the reported 50-learner, 10k-interaction regime.

---

## 7. Usage

### 7.1 Rule-based calculator

Open `Implement_a_rule_based_AI_for_Vedic_mathematics.ipynb` and run all cells. The final cell launches:

```
--- Vedic Mathematics AI ---
1. Urdhva Tiryagbhyam
2. Nikhilam Sutra
3. Yavadunam
```

Programmatic API:

```python
from pathlib import Path  # or copy the three functions into a module
product = urdhva_tiryagbhyam(123, 456)
product = nikhilam_sutra(97, 98)
square  = yavadunam_sutra_refined(102)
```

### 7.2 Research pipeline

Execute the hybrid notebook top-to-bottom. Phase order is mandatory: corpus → features → RF fit → student simulation → tag cleaning → NNKT training. Random seeds are not pinned in the original cells; set `random.seed` / `np.random.seed` / `tf.random.set_seed` if exact numerical reproduction is required.

---

## 8. Limitations and Future Work

- Ground-truth sutra assignment uses a single $\delta$-ball heuristic; alternative pedagogies (e.g., *Anurupyena*, *Ekadhikena Purvena*) are not labelled.
- Nikhilam remains a rare class under uniform operand sampling; stratified or importance sampling would yield a more balanced RF training set.
- The tracer is a single-layer LSTM; stacked LSTMs with dropout and auxiliary features (time since last attempt, consecutive errors, item difficulty) are proposed but not trained in the current notebooks.
- All learners are synthetic. Transfer to classroom click-stream data is untested.

---

## 9. Citation

If this repository supports your research, please cite it as:

```bibtex
@misc{vedic-math-hybrid-kt,
  title        = {Enhancing Vedic Mathematics Education: A Hybrid Approach
                  Using Random Forest for Question Classification and
                  Neural Network-Based Knowledge Tracing in the Presence
                  of Noisy Skill Tags},
  year         = {2026},
  note         = {Research prototype and accompanying notebooks},
  howpublished = {\url{https://github.com/<user>/vedic-math-hybrid-kt}}
}
```

---

## 10. Licence

This project is released under the MIT Licence unless otherwise noted in individual notebooks. Vedic sutra names and classical statements are historical/public-domain formulations; the software implementations and experimental design are original to this repository.

---

*Correspondence regarding the experimental protocol, feature definitions, or Focal-Loss hyperparameters may be opened as repository issues.*
