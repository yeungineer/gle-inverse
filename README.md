# GLE Inverse

Code for *History-Conditioned Inference of Memory and Stochastic Forcing in Generalized Langevin Equations under Lévy Noise*. All data are simulated.

Use Python 3.10. Plotting needs LaTeX with `amsmath`, `amssymb` and `bm`.

```bash
pip install numpy==2.2.6 scipy==1.15.3 torch==2.13.0 \
  matplotlib==3.10.9 PyYAML==6.0.3 psutil==7.2.2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
```

The model, numerical methods and baselines are in `model/`. Run a basic experiment from the repository root:

```bash
python run.py --config brownian.yaml --output results/Brownian
python run.py --plot results/Brownian
```

Use `brownian_levy.yaml` for Lévy noise. Each `experiments/figureX_*.py` contains that figure's parameters and seeds, calculation (`run`), and plotting (`plot`).

Replace `1` below with any figure number from 1 to 7. Choose a new output directory for each computation.

```bash
python experiments/run.py 1 --output results/Figure1_new
python experiments/run.py 1 --plot --input results/Figure1_new
# Redraw the supplied results without recomputing:
python experiments/run.py 1 --plot --output results/Figure1_redrawn
```

Saved paper data and figures are in `results/Figure1`–`Figure7`. New runs save parameters, arrays and metrics; plotting reads them and saves PDF/PNG. All figures share `model/style.py`.
