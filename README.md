# GLE Inverse

This repository accompanies *History-Conditioned Inference of Memory and Stochastic Forcing in Generalized Langevin Equations under Lévy Noise*. It implements a history-conditioned Kramers–Moyal inverse framework for separating memory drift, state-dependent Brownian fluctuations, and optional symmetric α-stable Lévy jumps, with the memory kernel recovered through RKHS-regularized Volterra inversion.

## Install

Python 3.10 or later is required.

```bash
python3 -m pip install -r requirements.txt
```

## Main workflow

```bash
python3 run/run_kramers_moyal.py
python3 run/run_kramers_moyal.py --config run/configs/kramers_moyal_brownian_levy.yaml
```

## Paper figures

Figures 1–7 each have a runner in `paper/code_figure_<n>/`.

Generate data and plot a figure:

```bash
python3 paper/code_figure_1/run_figure_1.py
```

Plot an existing run:

```bash
python3 paper/code_figure_1/run_figure_1.py \
  --mode plot \
  --run-dir paper/code_figure_1/output/<run-id>
```

The seeds used by each figure are listed directly in its `figure_<n>.yaml` file.

Each run keeps only four files:

```text
figure_<n>_data.npz
figure_<n>_metadata.yaml
Figure_<n>.pdf
Figure_<n>.png
```

Generation intermediates are temporary and are deleted before the run finishes.
