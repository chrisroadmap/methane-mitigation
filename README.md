# How much methane removal is required to avoid overshooting 1.5°C?

This repository reproduces the results from Smith & Mathison, submitted.

## Reproduction steps

### Set up conda repository

This assumes that you are using `anaconda` and `python`. Currently, `fair` and `fair-calibrate` appear to be most stable with `python` versions 3.8, 3.9, 3.10 and 3.11. Others may work, but these ones are tested.

1. Create your environment:

```
$ conda env create -f environment.yml
```
2. If you want to make nice version-control friendly notebooks, which will remove all output and data upon committing, run

```
$ nbstripout --install
```

### Run and reproduce results

1. Fire up jupyter notebook

```
$ jupyter notebook
```

2. In `notebook`, navigate to `notebooks` directory. Run the notebooks in this order:
  - `adaptive-removal-1.4.0.ipynb`: this does the data crunching. It will likely take between 6 and 24 hours, depending on your machine.
  - `zec-1.4.0.ipynb`: calculate ZEC and carbon cycle metrics
  - `analyse-1.4.0.ipynb`: produce the results and plots reported in the paper
