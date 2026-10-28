"""Cleaning, EDA, RFM, cohorts, channel ROI, forecasting. Run after 01_generate_data.py."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
C, T = "outputs/charts/", "outputs/tables/"
log = {}

# ============ 1. LOAD & CLEAN ============
cust = pd.read_csv("data/raw/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/raw/products.csv")
tx = pd.read_csv("data/raw/transactions.csv", parse_dates=["order_date"])
cac = pd.read_csv("data/raw/channel_costs.csv")

n0 = len(tx); tx = tx.drop_duplicates(subset="line_id"); log["duplicate_lines_removed"] = n0 - len(tx)
cust["region"] = cust["region"].str.strip().str.title(); log["region_labels_after_cleaning"] = cust["region"].nunique()
bad_age = (cust.age > 100) | (cust.age < 16); cust.loc[bad_age, "age"] = np.nan
log["age_outliers_fixed"] = int(bad_age.sum()); log["age_missing_imputed"] = int(cust.age.isna().sum())
cust["age"] = cust["age"].fillna(cust["age"].median())

df = tx.merge(prod[["product_id", "category", "unit_cost"]], on="product_id").merge(cust[["customer_id", "region", "channel", "age", "signup_date"]], on="customer_id")
df["gross"] = df.quantity * df.unit_price * (1 - df.discount_pct)
df["cogs"] = df.quantity * df.unit_cost
df["valid"] = df.status.eq("Delivered")          # net revenue = delivered orders only
df["month"] = df.order_date.dt.to_period("M")
v = df[df.valid].copy(); v["profit"] = v.gross - v.cogs
v.to_csv("data/processed/clean_transactions.csv", index=False)

# ============ 2. KPIs ============
orders = v.groupby("order_id").agg(rev=("gross", "sum"), cust=("customer_id", "first"), date=("order_date", "first"))
oc = orders.groupby("cust").size()
kpi = {"Net revenue": v.gross.sum(), "Gross profit": v.profit.sum(), "Gross margin %": 100 * v.profit.sum() / v.gross.sum(),
       "Delivered orders": len(orders), "Active customers": orders.cust.nunique(), "Avg order value": orders.rev.mean(),
       "Repeat purchase rate %": 100 * (oc > 1).mean(),
       "Cancel rate %": 100 * df.drop_duplicates("order_id").status.eq("Cancelled").mean(),
       "Return rate %": 100 * df.drop_duplicates("order_id").status.eq("Returned").mean()}
pd.Series(kpi).round(2).to_csv(T + "kpis.csv", header=["value"])

# ============ 3. TRENDS ============
m = v.groupby("month").agg(revenue=("gross", "sum"), profit=("profit", "sum"), orders=("order_id", "nunique"))
m.index = m.index.to_timestamp()
fig, ax = plt.subplots(figsize=(10, 4.5)); ax.bar(m.index, m.revenue, width=20, color="#4C78A8", label="Revenue")
ax2 = ax.twinx(); ax2.plot(m.index, m.orders, color="#F58518", marker="o", label="Orders"); ax2.grid(False)
ax.set_title("Monthly net revenue & orders — Nov/Dec peak is clearly visible"); ax.set_ylabel("Revenue ($)"); ax2.set_ylabel("Orders")
fig.tight_layout(); fig.savefig(C + "01_monthly_revenue.png"); plt.close()
peak = m.revenue.idxmax(); yoy = m.loc["2024"].revenue.sum() / m.loc["2023"].revenue.sum() - 1

# ============ 4. CATEGORY / PRODUCT / REGION / DISCOUNT ============
cat = v.groupby("category").agg(revenue=("gross", "sum"), profit=("profit", "sum")); cat["margin_pct"] = 100 * cat.profit / cat.revenue
cat = cat.sort_values("revenue"); cat.round(1).to_csv(T + "category_performance.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); ax[0].barh(cat.index, cat.revenue, color="#4C78A8"); ax[0].set_title("Revenue by category")
ax[1].barh(cat.index, cat.margin_pct, color="#54A24B"); ax[1].set_title("Gross margin % by category"); fig.tight_layout(); fig.savefig(C + "02_category.png"); plt.close()

top = v.groupby("product_id").gross.sum().sort_values(ascending=False); top10_share = top.head(10).sum() / top.sum()
share = top.cumsum() / top.sum(); n80 = int((share < .8).sum() + 1)
fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(range(1, len(share) + 1), share.values * 100, color="#E45756"); ax.axhline(80, ls="--", c="grey")
ax.set_title(f"Pareto: {n80} of {len(share)} products drive 80% of revenue"); ax.set_xlabel("Products (ranked)"); ax.set_ylabel("Cumulative revenue %"); fig.tight_layout(); fig.savefig(C + "03_pareto.png"); plt.close()

v["disc_band"] = pd.cut(v.discount_pct, [-.01, 0, .05, .10, .20], labels=["0%", "5%", "10%", "20%"])
disc = v.groupby("disc_band", observed=True).agg(profit=("profit", "sum"), rev=("gross", "sum")); disc["margin_pct"] = 100 * disc.profit / disc.rev
disc.round(1).to_csv(T + "discount_vs_margin.csv")

# ============ 5. RFM SEGMENTATION ============
snap = v.order_date.max() + pd.Timedelta(days=1)
rfm = v.groupby("customer_id").agg(recency=("order_date", lambda s: (snap - s.max()).days), frequency=("order_id", "nunique"), monetary=("gross", "sum"))
rfm["R"] = pd.qcut(rfm.recency.rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F"] = pd.qcut(rfm.frequency.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M"] = pd.qcut(rfm.monetary.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
def seg(r):
    if r.R >= 4 and r.F >= 4: return "Champions"
    if r.F >= 3 and r.R >= 3: return "Loyal"
    if r.R <= 2 and r.F >= 3: return "At Risk (high value)"
    if r.R >= 4 and r.F <= 2: return "New / Promising"
    if r.R <= 2 and r.F <= 2: return "Hibernating"
    return "Needs Attention"
rfm["segment"] = rfm.apply(seg, axis=1); rfm.to_csv("data/processed/rfm_scores.csv")
sg = rfm.groupby("segment").agg(customers=("monetary", "size"), revenue=("monetary", "sum"))
sg["cust_pct"] = 100 * sg.customers / sg.customers.sum(); sg["rev_pct"] = 100 * sg.revenue / sg.revenue.sum(); sg = sg.sort_values("rev_pct", ascending=False)
sg.round(1).to_csv(T + "rfm_segments.csv")
fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(len(sg)); ax.bar(x - .2, sg.cust_pct, .4, label="% of customers", color="#9ecae9"); ax.bar(x + .2, sg.rev_pct, .4, label="% of revenue", color="#4C78A8")
ax.set_xticks(x); ax.set_xticklabels(sg.index, rotation=20, ha="right"); ax.legend(); ax.set_title("RFM segments: share of customers vs share of revenue"); fig.tight_layout(); fig.savefig(C + "04_rfm.png"); plt.close()
champ = sg.loc["Champions"] if "Champions" in sg.index else None
risk = sg.loc["At Risk (high value)"] if "At Risk (high value)" in sg.index else None

# ============ 6. COHORT RETENTION ============
o = orders.reset_index(); o["m"] = o.date.dt.to_period("M"); first = o.groupby("cust").m.min().rename("cohort")
o = o.join(first, on="cust"); o["age_m"] = (o.m - o.cohort).apply(lambda x: x.n)
coh = o.groupby(["cohort", "age_m"]).cust.nunique().unstack(); ret = coh.div(coh[0], axis=0) * 100
ret = ret[coh[0] >= 40].iloc[:, :13]; ret.round(1).to_csv(T + "cohort_retention.csv")
fig, ax = plt.subplots(figsize=(10, 6)); im = ax.imshow(ret.values, cmap="Blues", vmin=0, vmax=40, aspect="auto")
ax.set_xticks(range(ret.shape[1])); ax.set_yticks(range(len(ret))); ax.set_yticklabels([str(i) for i in ret.index]); ax.grid(False)
ax.set_xlabel("Months since first order"); ax.set_title("Customer retention by acquisition cohort (%)"); plt.colorbar(im); fig.tight_layout(); fig.savefig(C + "05_cohorts.png"); plt.close()
m1, m3, m6 = ret[1].mean(), ret[3].mean(), ret[6].mean()

# ============ 7. CHANNEL ROI (LTV : CAC) ============
cp = v.groupby("customer_id").profit.sum()
ch = cust.set_index("customer_id").join(cp.rename("profit")).fillna({"profit": 0}).groupby("channel").agg(customers=("profit", "size"), ltv=("profit", "mean"))
ch = ch.join(cac.set_index("channel")); ch["ltv_cac"] = ch.ltv / ch.cac; ch = ch.sort_values("ltv_cac"); ch.round(2).to_csv(T + "channel_roi.csv")
fig, ax = plt.subplots(figsize=(8, 4)); ax.barh(ch.index, ch.ltv_cac, color=["#E45756" if x < 3 else "#54A24B" for x in ch.ltv_cac]); ax.axvline(3, ls="--", c="grey")
ax.set_title("LTV:CAC by acquisition channel (3x benchmark)"); fig.tight_layout(); fig.savefig(C + "06_channel_roi.png"); plt.close()

# ============ 8. FORECAST (log-trend x monthly seasonality; 2023 was a ramp-up so growth is concave) ============
def fit_forecast(series, horizon):
    y = np.log(series.values); t = np.arange(len(y)); x = np.log(t + 1); b, a = np.polyfit(x, y, 1)
    resid = pd.Series(y - (a + b * x), index=series.index); seas = resid.groupby(resid.index.month).mean()
    fut = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    pred = [np.exp(a + b * np.log(len(y) + i + 1) + seas.get(d.month, 0)) for i, d in enumerate(fut)]
    return pd.Series(pred, index=fut), resid.std()
rev = m.revenue
errs = []                                                          # rolling-origin backtest (3-month horizon)
for end in range(18, 22):
    tr, te = rev.iloc[:end], rev.iloc[end:end + 3]; p, _ = fit_forecast(tr, len(te))
    errs.append(np.mean(np.abs(p.values - te.values) / te.values))
mape = 100 * np.mean(errs)
fc, sd = fit_forecast(rev, 6)
fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(rev.index, rev, label="Actual", color="#4C78A8", marker="o"); ax.plot(fc.index, fc, label="Forecast", color="#F58518", marker="o", ls="--")
ax.fill_between(fc.index, fc * np.exp(-1.96 * sd), fc * np.exp(1.96 * sd), color="#F58518", alpha=.2); ax.legend()
ax.set_title(f"6-month revenue forecast (rolling backtest MAPE {mape:.1f}%)"); fig.tight_layout(); fig.savefig(C + "07_forecast.png"); plt.close()
fc.round(0).to_csv(T + "forecast_next_6_months.csv", header=["forecast_revenue"])

# ============ 9. WRITE REPORT ============
rep = f"""# Executive Summary — E-Commerce Performance Analysis (2023–2024)

## Data quality actions
- Removed **{log['duplicate_lines_removed']}** duplicate transaction lines; standardised region labels to **{log['region_labels_after_cleaning']}** clean values
- Fixed **{log['age_outliers_fixed']}** impossible ages and imputed **{log['age_missing_imputed']}** missing ages (median)
- Net revenue is defined as **delivered orders only** (cancelled and returned orders excluded)

## Headline KPIs
""" + "\n".join(f"- {k}: **{v_:,.1f}**" for k, v_ in kpi.items()) + f"""

## Key findings
1. **Seasonality:** peak month was **{peak:%B %Y}**; revenue grew **{yoy:.0%} YoY** (2024 vs 2023).
2. **Category mix:** `{cat.index[-1]}` leads on revenue, but `{cat.margin_pct.idxmax()}` has the best margin ({cat.margin_pct.max():.0f}%) — `{cat.margin_pct.idxmin()}` the worst ({cat.margin_pct.min():.0f}%).
3. **Concentration:** top 10 products = **{top10_share:.0%}** of revenue; **{n80}** of {len(share)} products generate 80%.
4. **Discounting:** margin falls from **{disc.margin_pct.iloc[0]:.0f}%** (no discount) to **{disc.margin_pct.iloc[-1]:.0f}%** at 20% off.
5. **RFM:** Champions are **{champ.cust_pct:.0f}%** of customers but **{champ.rev_pct:.0f}%** of revenue; **{risk.customers:.0f}** high-value customers are At Risk (win-back target).
6. **Retention:** average cohort retention is **{m1:.0f}%** at month 1, **{m3:.0f}%** at month 3, **{m6:.0f}%** at month 6.
7. **Channel ROI:** best LTV:CAC = `{ch.ltv_cac.idxmax()}` ({ch.ltv_cac.max():.1f}x), worst = `{ch.ltv_cac.idxmin()}` ({ch.ltv_cac.min():.1f}x).
8. **Forecast:** next 6 months projected at **${fc.sum():,.0f}** (rolling backtest MAPE **{mape:.1f}%**; treat as directional).

## Recommendations
- Shift acquisition budget from `{ch.ltv_cac.idxmin()}` toward `{ch.ltv_cac.idxmax()}` and other channels above the 3x LTV:CAC benchmark.
- Launch a win-back campaign for the At Risk segment before the Nov–Dec peak; cap blanket 20% discounts which erode margin.
- Build a month-1 onboarding flow to lift early retention, the steepest drop-off point.
- Stock up on top-revenue SKUs ahead of Q4 using the forecast.
"""
open("outputs/REPORT.md", "w").write(rep); print(rep)
