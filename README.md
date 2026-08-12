# Bayesian Neural Networks for Stock Price Forecasting

Code, data and results for the project report *"Bayesian Neural Networks for
Stock Price Forecasting Across Three Financial Crises: A Calibration Audit
and Extension."*

Probabilistic Modelling, Leuphana University Lüneburg, 2026.
Shreyas Krishnamurty (4006526), Alwala Raghavendra Goud (4006339),
Ranjit Singh (4006230).

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

`results/` holds the complete 24-experiment run under the final protocol:
re-anchored Middle East windows, multi-horizon Bayes-LR, heteroskedastic
noise and the adaptive temperature ladder. Every figure and number in the
report comes from this run. The statistical analysis reproduces in seconds:

```bash
python run_significance.py results figures/report
```

This prints the cross-experiment battery and writes
`figures/report/significance_tests.png`. Expected output: Friedman
chi-square 66.5 (p approximately 5e-13), mean ranks Bayes-LR 1.50,
FNN-Adam 2.67, MC-Dropout 3.29, BNN 3.42, FNN-SGD 4.88, LSTM 5.25, and
stock-level mean differences of +0.0059 (CBA.AX), +0.0012 (DAI.DE),
+0.0036 (MMM) and +0.0487 (600118.SS).

Headline numbers from this run, for reference:

| Metric | Value |
|---|---|
| BNN coverage at nominal 95% | 86.1% |
| BNN ECE | 0.099 |
| BNN coverage after temperature scaling | 92.5% |
| Bayes-LR beats BNN on pooled RMSE | 23 of 24 experiments |
| Weakest crisis (Middle East) coverage | 80.4% |
| Worst single experiment (600118.SS, M2) | 66.6% |
| Swap acceptance after the ladder fix | 26.0% (range 23.0-31.5%) |

Two things this run does not contain. First, the faithful-replication
baseline quoted in Table 2 of the report (76.9% coverage, ECE 0.190):
reproducing it means disabling both fixes and re-running the study, which the
report states explicitly rather than presenting as a matched ablation. To
generate it:

```bash
python run.py --bnn-homoskedastic --fixed-ladder
```

Second, the per-day forecast errors and interval-hit sequences that the
Diebold-Mariano, Kupiec and Christoffersen tests need. `run.py` saves those
now, but this run predates that change, so `run_significance.py` reports the
gap and runs the aggregate tests only.

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

A complete run produces roughly 400 figures. This repository ships a curated
subset in four non-overlapping folders, all from the single run saved in
`results/`, so they are mutually consistent and match the report.

- `figures/report/` - the nine figures that appear in the report, in figure
  order: `price_overview` (Fig. 1), `fan_chart_MMM_R1` (Fig. 2),
  `calibration_MMM_R1` (Fig. 3), `recalibration_MMM_R1` (Fig. 4),
  `significance_tests` (Fig. 5), `multistep_Bayes-LR_MMM_M1` (Fig. 6),
  `combined_trace_MMM_R1` (Fig. 7), `burnin_diagnostic_MMM_R1` (Fig. 8) and
  `multistep_fan_BNN_MMM_M1` (Fig. 9).
- `figures/summary/` - cross-experiment results: ECE ranking, accuracy versus
  calibration, coverage by crisis type, recalibration summary, per-setup
  calibration grids, and the two conceptual explainer plots.
- `figures/calibration_all/` - the calibration, recalibration, RMSE and
  volatility-split figures for every experiment. These carry the project's
  central claim, so they are included in full.
- `figures/worked_examples/` - the remaining figure types for three
  representative experiments: MMM/R1 (COVID), MMM/M1 (Middle East escalation)
  and CBA.AX/M1 (the hardest case).

Two of the report figures are not produced by `run.py` directly:
`price_overview.png` was drawn by a small standalone script for the report,
and `significance_tests.png` comes from `run_significance.py`. Every other
figure regenerates when `run.py` is re-run.

## Main findings

- The faithfully replicated BNN is materially overconfident: nominal 95%
  intervals contain the true price 76.9% of the time on average.
- Two independent causes: a homoscedastic noise term that cannot react to
  volatility regimes, and a parallel-tempering sampler whose chains swapped
  only 11.7% of the time.
- Fixing both raises average coverage to 86.1% and lowers expected
  calibration error to 0.099; post-hoc temperature scaling lifts coverage to
  92.5%. The calibration gain costs point accuracy, the expected
  sharpness-versus-calibration trade-off. Calibration stays regime-dependent:
  80.4% in the Middle East setups against 88-90% elsewhere.
- A Bayesian linear regression with about seven parameters per horizon beats
  the BNN on accuracy in 23 of 24 experiments (Wilcoxon p is approximately
  8e-07; Friedman across six models p is approximately 5e-13). Because the
  experiments share stocks and overlapping windows, the strictest evidence is
  the stock-level cluster analysis, where all four clusters agree in sign.
  The calibration comparison between the two is not significant (14 of 24,
  p = 0.076).

## Troubleshooting

If a run finishes but the expected figures or the significance step are
missing:

```bash
python diagnose_significance.py
```

It checks, in the order `run.py` does, every condition that has to hold:
whether `scipy` and the local modules import, how many result files exist,
which models are present in which experiments, whether the per-day data is
there, which figures are missing, and then runs the significance step and
prints the real traceback if it fails. It locates `outputs/` itself; if it
cannot, pass the path:

```bash
python diagnose_significance.py path/to/outputs/results
```

The most common cause is that `run.py` writes to `outputs/` relative to the
working directory it was launched from. In an IDE this is often not the
project folder, so the output ends up somewhere unexpected rather than
missing.

One caveat when running the diagnostic inside this repository: its figure
check expects the flat `outputs/figures/` layout that `run.py` produces, so
against the curated `figures/summary`, `figures/calibration_all` and
`figures/worked_examples` folders here it will report most figures as
missing. That is expected; the check is meaningful against a real run's
output directory.

## Requirements

Python 3.10+, with `numpy`, `pandas`, `scikit-learn`, `matplotlib`,
`tensorflow`, `pymc` and `scipy` (see `requirements.txt`). PyMC's
multiprocessing can be slow on Windows; if sampling stalls, try running with
fewer chains.
