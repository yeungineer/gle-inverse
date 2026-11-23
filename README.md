# GLE Inverse

## 安装依赖

```bash
python -m pip install "numpy>=2.0" scipy matplotlib torch PyYAML
```

## 运行反演

在项目根目录运行布朗或布朗–Lévy 算例。

```bash
python -B run.py --config brownian.yaml
python -B run.py --config brownian_levy.yaml
```

结果保存在 `results/`。单次反演可用 `python -B run.py --plot 结果目录` 重绘。

## 论文图 1–8

按图号计算并绘图，结果保存在 `results/FigureN/`。`--workers` 指定并行进程数。

```bash
python -B experiments/run.py 1 2 3 4 5 6 7 8 --workers 4
```

用已有数据重绘图 1 和图 2。

```bash
python -B experiments/run.py 1 2 --plot
```
