# Do Shoppers Follow the Catalogue?
### Product Communities, Network Analysis and Machine Learning from Instacart Baskets

**Team:** Syed Wassi Ul Haque · Nicolas Ignacio Rivolta
**Course:** Data Science (CITHN1007), TUM Campus Heilbronn, WS 2026/27
**Status:** Proposal submitted (Milestone 1, Nov 4). Next: Milestone 2, analysis progress report, **Dec 16, 2026**.

> This file is the single source of truth for the project's content: the question, the data and the plan.
> It is written for the team and for Claude Code. If a decision changes, update this file and the decision log.

---

## 1. The project in one paragraph

Supermarkets and online grocery apps organise products by aisles and departments (dairy, produce, pantry…). Customers, however, shop by habits and occasions ("missions"): a pasta dinner needs spaghetti (dry goods), tomato sauce (pantry), parmesan (dairy) and basil (produce), so four departments serve one idea. We build a **co-purchase network** of products from real Instacart baskets, detect **communities** of products that are bought together, and test whether these communities match the catalogue or capture shopping behaviour better. We then use the communities to solve a concrete problem: **which N products should a small-format store (mini-market) stock** to serve the most baskets. Two machine-learning questions test whether communities carry useful information for prediction and customer segmentation.

---

## 2. Main question and sub-questions

**Main question:** Do co-purchase product communities outperform catalogue categories in capturing real shopping behaviour, and can those communities guide assortment decisions in small-format retail?

The six sub-questions form a chain: **find → test → explain → apply → predict → segment**.

| # | Sub-question | Type | Lead |
|---|---|---|---|
| SQ1 | Which communities form via Louvain detection, and how closely do they match aisles and departments? | Network + statistical test | Syed |
| SQ2 | Do baskets span fewer groups under community grouping than under department grouping? | Evaluation | Nicolas |
| SQ3 | Which products are most central (degree, closeness, betweenness)? | Network | Syed |
| SQ4 | For a mini-market stocking N products, which strategy (popularity / centrality / community) covers the most baskets? | Evaluation / application | Nicolas |
| SQ5 [ML] | Does community membership improve reorder prediction over department membership? | Supervised ML | Syed |
| SQ6 [ML] | Do K-means clusters of users by community-purchase vectors reveal interpretable shopper segments? | Unsupervised ML | Nicolas |

**Priority if time runs short:** SQ1, SQ2 and SQ4 answer the main question (core). SQ3 supports SQ4. SQ5 and SQ6 are the ML extension.

---

## 3. Audience

- **Primary:** online grocery catalogue and recommendation designers (e.g. "shop by mission" sections, recommending the rest of a mission) and supermarket layout planners.
- **Secondary:** convenience / small-format retail operators choosing an assortment under space constraints.
- **Data story (PacificVis contest, optional):** anyone who buys groceries. Message: *"You don't shop by department, you shop by mission."*

---

## 4. Data

**Source:** Instacart Online Grocery Basket Analysis Dataset (public, Kaggle; released by Instacart in 2017).
Link: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

### 4.1 Files

| File | One row is | Rows | Key columns | Used for |
|---|---|---|---|---|
| `orders.csv` | one order | 3,421,083 | order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order | user IDs (split), order context, user history |
| `order_products__train.csv` | one product in a user's **last** order | 1,384,617 (131,209 orders) | order_id, product_id, add_to_cart_order, reordered | baskets, network, evaluation, SQ5 |
| `order_products__prior.csv` | one product in a user's **earlier** orders | ~32.4 million | same as train | SQ6 (user history), optional larger network |
| `products.csv` | one product | 49,688 | product_id, product_name, aisle_id, department_id | names, catalogue structure |
| `aisles.csv` | one aisle | 134 | aisle_id, aisle | catalogue grouping (fine) |
| `departments.csv` | one department | 21 | department_id, department | catalogue grouping (coarse) |

Joins: `orders ←order_id→ order_products ←product_id→ products → aisles, departments`.

### 4.2 Key facts (verified on the files)

- 206,209 users; each has 4–100 orders.
- Median basket size **9** products (mean 10.6).
- **59.9%** of basket items are reorders; for users on their 46th order it is 81.1%. Buying is habitual, so co-purchase patterns are stable.
- Produce is ~30% of all purchased items, dairy & eggs ~17%: produce will dominate the network.
- **Top 1,000 products cover 52.6% of all basket lines** (each was bought at least 231 times in the train file). Top 5,000 cover 80.4%. *(The submitted proposal said "≈80%" for the top 1,000; this is wrong and must be corrected in Milestone 2.)*
- ~25,000 of the 39,123 products appearing in train baskets were bought fewer than 10 times.

### 4.3 Quirks and limits

- **`reordered`** = 1 if this user bought this product in **any** earlier order (not only the previous one).
- **`eval_set`:** `prior` = history, `train` = last order with known products, `test` = last order with hidden products (no labels; not usable).
- **`days_since_prior_order`** is empty for each user's first order and **capped at 30** ("30 or more").
- **`order_dow`** is coded 0–6; Instacart never documented which code is which day. 0–1 are the busiest and probably the weekend; treat this as an assumption.
- **Aisles and departments are catalogue categories**, not physical shelves (Instacart is an online service).
- No prices, store locations, calendar dates or demographics. US customers only.
- Licence/credit: cite the Kaggle page and Instacart.

---

## 5. Core concepts

- **Co-purchase network:** nodes = products; an edge connects two products bought together more often than chance.
- **Lift:** `lift(A,B) = P(A and B) / (P(A) · P(B))`. Lift 1 = no link; lift 3 = three times more often than chance. Example from the data: banana + organic strawberries appear together in 2,174 baskets but lift is only 1.4 (both are everywhere); spaghetti + marinara sauce appear together in only 37 baskets but lift is 14.8 (a real mission).
- **Minimum co-purchase count:** pairs must co-occur in at least k baskets (e.g. 15) so rare pairs cannot get a large lift by luck.
- **Community (Louvain):** groups of products densely connected inside and sparsely between; Louvain maximises **modularity Q** (edges inside groups minus the share expected by chance). The **resolution** parameter controls the number of communities.
- **NMI (normalised mutual information):** 0–1 similarity between two groupings of the same products (communities vs departments/aisles).
- **Permutation test:** shuffle the department labels many times; if the real NMI is far above all shuffled NMIs, the match is not chance.
- **Centrality:** degree (number of partners → anchors), closeness (few steps to everyone), betweenness (lies on shortest paths between others → bridges between missions).
- **Basket coverage:** share of a basket's products that a store stocks; a basket is **served** if coverage ≥ 80%.

**Pilot result (top 250 products, train baskets):** 8 communities, modularity 0.51, NMI with departments 0.34 vs 0.10 on average for 1,000 shuffles (max 0.14), so p < 0.001. Communities follow the catalogue more than chance but only partly. To be repeated on the full setup.

---

## 6. Planned analysis (step by step)

All steps use **pandas, NumPy, SciPy, NetworkX, scikit-learn, matplotlib/seaborn**.

### Step 0. Data preparation
1. Load the six CSVs from `data/` (relative path, never a personal path).
2. Merge baskets with products, aisles, departments; count rows before and after each merge.
3. **Split users 80/20** (`GroupShuffleSplit`, group = user_id, fixed seed). Build everything on the 80%; evaluate SQ2, SQ4, SQ5 on the 20%. This is a split by **user**, separate from Instacart's prior/train split.
4. Keep the **top 1,000 products** by number of training baskets (robustness check: top 2,000).

### Step 1. EDA
Basket size, product popularity (long tail), department shares, reorder rate, **cumulative coverage of top-N products**, **purchase counts per product**, **distribution of lift values**. These plots justify the top-N, the lift threshold and the minimum count.

### Step 2. Build the co-purchase network
- Baskets × products 0/1 sparse matrix `X`; co-occurrence `X.T @ X`; lift from counts.
- Keep edges with lift ≥ threshold and co-count ≥ k (start: lift ≥ 3.5, k ≥ 15; tune from EDA; second threshold for robustness).
- Report nodes, edges, density, giant component; compare clustering and path length with a random (Erdős–Rényi) graph of the same size.

### SQ1. Communities vs catalogue
- `nx.community.louvain_communities(G, resolution=r, seed=s)`; modularity.
- NMI (and ARI) of communities vs departments and vs aisles.
- Permutation test: 1,000 shuffles of department labels → p-value.
- Describe each community by its top products and its department mix; name the missions.
- Risk check: flag if one community holds > 40% of products (produce blob).

### SQ2. Groups per basket
- For each held-out basket, count distinct groups among its top-N products under: communities, departments, random grouping with the same group sizes (repeat 100×).
- **Fair comparison:** tune Louvain resolution to ≈ 21 communities (same as departments).
- Paired Wilcoxon signed-rank test (community vs department on the same baskets); report median difference.

### SQ3. Centrality
- Degree, closeness, betweenness (`nx.*_centrality`).
- Compare each ranking with popularity rank (Spearman). Identify **anchors** (high degree) and **bridges** (high betweenness). These feed SQ4.

### SQ4. Mini-market assortment
- Strategies for choosing N products: **popularity** (baseline), **centrality**, **community-based** (each community gets shelf share proportional to its basket usage; inside it, most central products first).
- For N = 300, 500, 1,000: share of held-out baskets with ≥ 80% coverage (also full coverage).
- Output: coverage curves (one per strategy) → best strategy, sensible store size, product list.
- Popularity baseline already measured on held-out users: N=300 → 5.1% baskets served, N=500 → 8.8%, N=1,000 → 17.5% (items covered 34.9%, 41.9%, 52.6%).

### SQ5 [ML]. Does community help predict reorders?
- **Target:** `reordered` for items of top-N products in held-out baskets.
- **Shared features:** cart position (absolute and relative), basket size, order number, days since prior order, product popularity.
- **Models:** baseline (shared features) vs + department vs + community (same number of groups, ≈ 21), each with logistic regression and gradient boosting.
- **Metrics:** ROC-AUC, F1 (+ PR-AUC); confidence intervals by bootstrapping test users.
- **No leakage:** never use "has this user bought this product before" (that is the target's definition). Product-level statistics are computed on training users only.
- A null result (no gain) is informative.

### SQ6 [ML]. Shopper segments
- Needs **`order_products__prior.csv`** (one basket per user is too little).
- User vector = share of each user's purchases in each community (normalised to sum 1); comparison: department vectors with the same dimension.
- K-means for k = 5–10; choose k with elbow + silhouette + interpretability.
- Evaluate: stability across seeds (ARI), interpretability (name each segment), and whether segments differ in held-out behaviour (e.g. next basket composition, reorder rate). Silhouette alone is not a fair comparison across feature spaces.

---

## 7. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Different numbers of groups bias SQ2 and SQ5 | Fewer, bigger groups automatically mean fewer groups per basket / different information | Tune communities to ≈ 21; compare with equal-size random grouping |
| Popular products connect to everything | Raw counts make produce the centre of the network | Lift-weighted edges + minimum co-purchase count |
| Produce dominance → one giant community | Modularity and NMI would be misleading | Lift threshold; flag any community > 40% of products |
| Results depend on thresholds / seeds / top-N | One lucky setting is not a result | Repeat with a second threshold, several seeds, top 2,000 |
| Community features give no ML gain (SQ5) | Weak ML story | Null result is informative; report feature importance and CIs |
| US online data ≠ small-format stores | Overclaiming | Present store results as implications; the method transfers |

---

## 8. Timeline (updated Nov 24)

| Week | Tasks |
|---|---|
| Nov 24–30 | Repo + environment; data prep; fix EDA (top-N coverage, purchase counts, lift distribution) |
| Dec 1–7 | Network; SQ3 centrality; SQ1 Louvain + permutation test |
| Dec 8–15 | SQ2 groups per basket; SQ4 coverage curves; prepare SQ6 user vectors; Milestone 2 slides |
| **Dec 16** | **Milestone 2: analysis progress report** |
| Jan 6–20 | SQ5 reorder models (classification lecture Dec 21); SQ6 K-means (clustering lecture Jan 18) |
| Jan 13 | Story draft peer critique |
| Jan 20–25 | Robustness checks; final notebook runs top to bottom (**project complete Jan 25**) |
| Jan 27 / Feb 3 | Final presentation · optional PacificVis submission Feb 1 |

---

## 9. Course requirements that shape the work

- One **Jupyter notebook** must run **top to bottom from the raw data** and produce every figure shown.
- Each student is **questioned individually** on the whole analysis → both must understand every step.
- **AI policy:** AI tools only "as a sounding board or to prepare a write-up", not to create ideas or write the entirety of the code. A mandatory **AI-usage statement** is part of the report.

**Rules for Claude Code in this project:** explain concepts first, review and test the team's code, verify results independently, point out leakage and unfair comparisons. Do not write whole analysis sections for the team unless explicitly asked, and log any substantial AI help so it can go into the AI statement.
