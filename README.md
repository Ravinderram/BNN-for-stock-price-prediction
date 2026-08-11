# BNN for stock price prediction

Code and data for the project report *"Does the Confidence Hold? A Calibration
Audit and Methodological Extension of Bayesian Neural Networks for Stock Price
Forecasting Across Three Financial Crises."*

The project replicates the Bayesian neural network (BNN) of Chandra & He
(2021), extends its evaluation from one crisis to three, and tests something
the original paper never did: whether the model's stated confidence levels
actually hold. They do not. We trace the overconfidence to two causes, fix
both, and compare against five reference models.

Chandra, R., & He, Y. (2021). Bayesian neural networks for stock price
forecasting before and during COVID-19 pandemic. *PLOS ONE*, 16(7), e0253217.
https://doi.org/10.1371/journal.pone.0253217

## Quick start

```bash
pip install -r requirements.txt
python run.py                       # full study: 4 stocks x 6 setups
```

The full run is long (many hours: the BNN draws 100,000 MCMC samples on each
of 10 tempered chains per experiment, and the Bayesian linear regression fits
one NUTS model per forecast horizon). For a fast smoke test:

```bash
python run.py --quick --stocks MMM --setups R1
```

Useful flags: `--quick`, `--stocks`, `--setups`, `--no-pymc`,
`--bnn-homoskedastic` (disable the noise fix), `--fixed-ladder` (disable the
sampler fix). The last two exist to reproduce the "before" numbers.

Everything is written to `outputs/` **relative to the working directory you
launch from**, not to this folder. In an IDE, check the run configuration's
working-directory setting if you cannot find the output.

## What is in here

| Path | Contents |
|---|---|
| `run.py` | Orchestration: runs every experiment, saves results and figures, then runs the significance tests |
| `bnn.py` | The replicated BNN with Langevin-gradient parallel tempering, plus the heteroskedastic noise model and the adaptive temperature ladder |
| `pymc_reference.py` | Bayesian linear regression reference, one independent NUTS fit per forecast horizon |
| `baselines.py` | MC-Dropout, FNN (Adam and SGD), LSTM |
| `calibration.py` | Coverage, expected calibration error, temperature scaling |
| `data.py` | Loading, normalisation, and the six train/test splits |
| `plots.py` | All figures |
| `significance_tests.py` | Wilcoxon, Friedman, Holm correction, cluster analysis, Diebold-Mariano, Kupiec, Christoffersen |
| `run_significance.py` | Runs the whole test battery and writes the significance figure |
| `diagnose_significance.py` | Troubleshooting: reports what is missing and why |
| `*_clean.csv` | Daily closing prices, 2012-2026, for the four stocks |
| `results/` | Saved result files from two pipeline stages, plus summary tables |
| `figures/` | A curated subset of the output figures (see below) |

## Reproducing the analysis without re-running the study

`results/` holds saved result files from completed runs, so the statistical
analysis can be reproduced in seconds. There are two sets, corresponding to
the first two rows of Table 2 in the report:

| Folder | Pipeline stage | BNN coverage@95% | BNN ECE |
|---|---|---|---|
| `results/replication/` | Faithful replication (homoscedastic noise, fixed ladder) | 76.9% | 0.190 |
| `results/heteroskedastic/` | After the heteroskedastic noise fix, Eq. (6) | 82.9% | 0.131 |

The report's model comparison and significance analysis (Section 4.4,
Table 3, Figure 5) were computed on `results/heteroskedastic/`:

```bash
python run_significance.py results/heteroskedastic figures
```

This reproduces the reported figures exactly: Friedman chi-square 82.2
(p approximately 3e-16), mean ranks Bayes-LR 1.00, FNN-Adam 2.88, MC-Dropout
3.46, BNN 3.46, FNN-SGD 4.79, LSTM 5.42, and stock-level mean differences of
+0.0861, +0.0175, +0.0156 and +0.0419.

Running the same command on `results/replication/` gives different numbers
(Friedman chi-square 88.7; BNN mean rank 2.96 instead of 3.46). That is not
an inconsistency: the heteroskedastic fix widens the BNN's intervals and
costs roughly 17% in RMSE, which is exactly the sharpness-versus-calibration
trade-off the report documents, so the pre-fix BNN ranks better on RMSE alone.
Both sets are included so this trade-off can be inspected directly.

Neither set contains the per-day forecast errors and interval-hit sequences
that the Diebold-Mariano, Kupiec and Christoffersen tests need; `run.py`
saves those now, but both runs predate that change. The script reports this
rather than skipping silently, and runs the aggregate tests.

These runs also predate the re-anchoring of the Middle East windows described
in Section 3.1, so their M1/M2 rows correspond to the earlier window
definition. Re-running `run.py` regenerates everything under the current
protocol.

## Experimental design

Four stocks (3M, Commonwealth Bank of Australia, Daimler/Mercedes-Benz, China
Spacesat) crossed with six setups gives 24 experiments. Each setup trains only
on data preceding its test window.

| Setup | Crisis | Test window | Days |
|---|---|---|---|
| R1 | COVID-19 | 2020-02-01 to 2020-06-30 | 150 |
| R2 | COVID-19 | 2020-05-01 to 2020-06-30 | 60 |
| W1 | Russia-Ukraine | 2022-02-24 to 2022-12-31 | 310 |
| W2 | Russia-Ukraine | 2022-05-01 to 2022-12-31 | 244 |
| M1 | Middle East escalation | 2025-06-01 to 2026-05-31 | 364 |
| M2 | Middle East escalation | 2025-09-01 to 2026-05-31 | 272 |

The two setups per crisis differ in how much early-crisis data enters
training, which separates "the model has never seen a crisis" from "the model
cannot represent crisis behaviour at all."

## Figures

The complete run produces roughly 400 figures. This repository ships a curated
subset:

- `figures/summary/` - cross-experiment results: ECE ranking, accuracy versus
  calibration, coverage by crisis type, recalibration summary, per-setup
  calibration grids, and the two conceptual explainer plots.
- `figures/calibration_all/` - the calibration, recalibration, RMSE and
  volatility-split figures for every experiment. These carry the project's
  central claim, so they are included in full.
- `figures/worked_examples/` - every figure type for three representative
  experiments: MMM/R1 (COVID), MMM/M1 (Middle East escalation) and CBA.AX/M1
  (the hardest case), including the multi-horizon forecast and fan charts.

Provenance, since the folders come from different runs: `summary/` is from the
replication run (matching `results/replication/`), except
`significance_tests.png`, which was generated from `results/heteroskedastic/`
and therefore matches Figure 5 of the report. `calibration_all/` and
`worked_examples/` are from a later run under the corrected protocol with the
re-anchored Middle East windows and the multi-horizon Bayes-LR. Re-running
`run.py` regenerates all of them consistently from a single run.

## Main findings

- The faithfully replicated BNN is materially overconfident: nominal 95%
  intervals contain the true price 76.9% of the time on average, and far less
  in the hardest crisis combinations.
- Two independent causes: a homoscedastic noise term that cannot react to
  volatility regimes, and a parallel-tempering sampler whose chains swapped
  only 11.7% of the time.
- Fixing both, plus post-hoc temperature scaling, raises average coverage to
  90.3% and lowers expected calibration error from 0.190 to 0.106. The
  calibration gain costs roughly 17% in RMSE, which is the expected
  sharpness-versus-calibration trade-off.
- A Bayesian linear regression with about seven parameters per horizon beats
  the BNN on accuracy in all 24 experiments (Wilcoxon p is approximately
  6e-08; Friedman across six models p is approximately 3e-16). Because the
  experiments share stocks and overlapping windows, the strictest evidence is
  the stock-level cluster analysis, where all four clusters agree in sign.

## Requirements

Python 3.10+, with `numpy`, `pandas`, `scikit-learn`, `matplotlib`,
`tensorflow`, `pymc` and `scipy` (see `requirements.txt`). PyMC's
multiprocessing can be slow on Windows; if sampling stalls, try running with
fewer chains.
