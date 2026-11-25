# Do Shoppers Follow the Catalogue?

Product communities, network analysis and machine learning from Instacart baskets.
TUM Heilbronn · Data Science (CITHN1007) · WS 2026/27 · Syed Wassi Ul Haque & Nicolas Ignacio Rivolta

**Main question:** Do co-purchase product communities outperform catalogue categories (departments / aisles) in
capturing real shopping behaviour, and can those communities guide assortment decisions in small-format retail?
Full plan in [`PROJECT.md`](PROJECT.md).

## Get the data

Download the *Instacart Online Grocery Basket Analysis Dataset* from Kaggle
(https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) and put the six CSV files
in `data/` (see [`data/README.md`](data/README.md)). The data is not in git: the files are larger than GitHub's
100 MB limit.

## Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open `notebook/instacart_analysis.ipynb`. It must run top to bottom from the raw data and writes all figures to
`figures/` and all numbers to `results/`.

## Structure

```
notebook/   the analysis notebook (one section per step and sub-question)
src/        shared helper functions
data/       raw CSVs (not in git)
figures/    figures written by the notebook
results/    numbers written by the notebook (CSV / JSON)
```

## Working together

- `main` always runs. Nobody commits to it directly.
- Nico works on `nico-dev`, Syed on `syed-dev`.
- Pull often (`git pull origin main`), commit small steps, merge into `main` through a pull request.
- Notebooks conflict easily: clear outputs before committing, and avoid editing the same section at the same time.


