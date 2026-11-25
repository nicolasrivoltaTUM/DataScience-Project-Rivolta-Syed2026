# data/

Raw data goes here. Nothing in this folder is committed to git except this README.

Download the *Instacart Online Grocery Basket Analysis Dataset* from Kaggle:
https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

Put these six files directly in this folder (no subfolders, keep the names):

| File | Rows | What it is |
|---|---|---|
| `orders.csv` | 3,421,083 | one row per order: user, order number, day of week, hour, days since prior order, eval_set |
| `order_products__prior.csv` | 32,434,489 | products in each earlier ("prior") order. Can stay as `.csv.gz` |
| `order_products__train.csv` | 1,384,617 | products in each user's last order |
| `products.csv` | 49,688 | product name, aisle_id, department_id |
| `aisles.csv` | 134 | aisle names |
| `departments.csv` | 21 | department names |

Check: after copying, `ls data/` should show the six files above plus this README.
