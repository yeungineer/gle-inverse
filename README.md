# GLE Inverse

## Install dependencies

```bash
python -m pip install "numpy>=2.0" scipy matplotlib torch PyYAML
```

## Run an inversion

From the project root, run either the Brownian or the Brownian–Lévy case.

```bash
python -B run.py --config brownian.yaml
python -B run.py --config brownian_levy.yaml
```

Results are saved under `results/`. To replot a run, pass its output directory
to `python -B run.py --plot`.

## Reproduce Figures 1–8

Run all eight experiments with four worker processes. Each figure and its data
are saved in the corresponding directory under `results/`.

```bash
python -B experiments/run.py 1 2 3 4 5 6 7 8 --workers 4
```

Replot Figures 1 and 2 using the saved data.

```bash
python -B experiments/run.py 1 2 --plot
```
