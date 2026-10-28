"""Generate a realistic e-commerce dataset (2023-2024) with deliberate data-quality issues.
Swap this for a real dataset (e.g. Olist / UCI Online Retail) by keeping the same column names."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")
N_CUST = 4000

# ---------- customers ----------
channels = ["Organic", "Paid Search", "Social", "Email", "Referral"]
cust = pd.DataFrame({"customer_id": np.arange(1, N_CUST + 1)})
span = (END - START).days
cust["signup_date"] = START + pd.to_timedelta((rng.beta(1.3, 1.0, N_CUST) * span).astype(int), unit="D")
cust["region"] = rng.choice(["North", "South", "East", "West"], N_CUST, p=[.28, .24, .22, .26])
cust["channel"] = rng.choice(channels, N_CUST, p=[.30, .25, .20, .15, .10])
cust["age"] = rng.normal(34, 9, N_CUST).clip(18, 70).astype(int)

# ---------- products ----------
cats = {  # category: (price_lo, price_hi, cost_ratio, n_products)
    "Electronics": (40, 600, .80, 12), "Home & Kitchen": (15, 150, .60, 12),
    "Fashion": (12, 120, .50, 14), "Beauty": (8, 70, .40, 8),
    "Sports": (15, 200, .62, 8), "Books": (6, 35, .60, 6)}
rows, pid = [], 1
for c, (lo, hi, cr, n) in cats.items():
    for _ in range(n):
        p = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 2)
        rows.append((pid, f"{c.split(' ')[0]} Item {pid:02d}", c, p, round(p * cr * rng.uniform(.9, 1.1), 2)))
        pid += 1
prod = pd.DataFrame(rows, columns=["product_id", "product_name", "category", "unit_price", "unit_cost"])
pop = rng.zipf(1.6, len(prod)).astype(float).clip(max=30); pop /= pop.sum()

# ---------- orders (seasonality + channel-driven loyalty) ----------
month_w = {1:.85,2:.80,3:.90,4:.95,5:1.0,6:.95,7:.90,8:.95,9:1.0,10:1.1,11:1.5,12:1.7}
life_mult = {"Organic":1.0,"Paid Search":.8,"Social":.7,"Email":1.2,"Referral":1.4}
orders = []
oid = 1
for c in cust.itertuples():
    rate = rng.gamma(1.5, .6)
    t = c.signup_date + pd.Timedelta(days=int(rng.integers(0, 5)))
    active_end = t + pd.Timedelta(days=float(rng.exponential(300 * life_mult[c.channel])))
    first = True
    while t <= min(END, active_end):
        if first or rng.random() < month_w[t.month] / 1.7:
            orders.append((oid, c.customer_id, t.normalize())); oid += 1
        first = False
        t += pd.Timedelta(days=float(rng.exponential(60 / max(rate, .15))) + 1)
orders = pd.DataFrame(orders, columns=["order_id", "customer_id", "order_date"])
orders["status"] = rng.choice(["Delivered", "Cancelled", "Returned"], len(orders), p=[.92, .03, .05])
orders["payment_method"] = rng.choice(["Card", "UPI", "COD", "Wallet"], len(orders), p=[.40, .30, .18, .12])

lines = []
for o in orders.itertuples():
    k = rng.choice([1, 2, 3, 4], p=[.5, .3, .14, .06])
    for p in rng.choice(prod.product_id, k, replace=False, p=pop):
        d = rng.choice([0, .05, .10, .20], p=[.2, .2, .3, .3] if o.order_date.month == 11 else [.6, .15, .15, .10])
        lines.append((o.order_id, p, int(rng.choice([1, 2, 3], p=[.7, .22, .08])), d))
lines = pd.DataFrame(lines, columns=["order_id", "product_id", "quantity", "discount_pct"])
tx = (lines.merge(orders, on="order_id").merge(prod[["product_id", "unit_price"]], on="product_id")
      .sort_values(["order_id", "product_id"]).reset_index(drop=True))
tx.insert(0, "line_id", np.arange(1, len(tx) + 1))
tx = tx[["line_id","order_id","customer_id","order_date","product_id","quantity","unit_price","discount_pct","status","payment_method"]]

cac = pd.DataFrame({"channel": channels, "cac": [15, 60, 45, 20, 30]})

# ---------- inject data-quality issues (cleaned in step 2) ----------
dup = tx.sample(frac=.01, random_state=1); tx = pd.concat([tx, dup]).sample(frac=1, random_state=2).reset_index(drop=True)
idx = cust.sample(frac=.06, random_state=3).index
cust.loc[idx, "region"] = [s.lower() + " " if rng.random() < .5 else s.upper() for s in cust.loc[idx, "region"]]
cust["age"] = cust["age"].astype(float)
cust.loc[cust.sample(frac=.04, random_state=4).index, "age"] = np.nan
cust.loc[cust.sample(frac=.005, random_state=5).index, "age"] = 150

cust.to_csv("data/raw/customers.csv", index=False); prod.to_csv("data/raw/products.csv", index=False)
tx.to_csv("data/raw/transactions.csv", index=False); cac.to_csv("data/raw/channel_costs.csv", index=False)
print(f"customers={len(cust):,}  products={len(prod)}  orders={len(orders):,}  lines={len(tx):,}")
