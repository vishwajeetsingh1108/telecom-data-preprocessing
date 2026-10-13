"""
Week 2 - Data Acquisition, Cleaning and Preprocessing in Telecom
Simulated telecom customer dataset (churn-style) + full preprocessing workflow.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

rng = np.random.default_rng(42)
R = {}  # collected results for the report

# ---------------------------------------------------------------- 1. SOURCING
def generate_raw(n=5000):
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[.45, .35, .20])
    pay = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    base = np.where(internet == "Fiber optic", 80, np.where(internet == "DSL", 55, 25))
    monthly = np.round((base + rng.normal(0, 8, n)).clip(15), 2)
    data_gb = np.where(internet == "No", rng.gamma(2, 2, n),
                       rng.gamma(4, 18, n) * np.where(internet == "Fiber optic", 1.4, 0.8))
    calls = np.round(rng.gamma(5, 70, n), 1)
    sms = rng.poisson(40, n)
    support = rng.poisson(1.5, n)
    latency = np.round(rng.normal(45, 12, n).clip(10), 1)
    senior = rng.choice([0, 1], n, p=[.84, .16])
    gender = rng.choice(["Male", "Female"], n)
    logit = (-1.2 - 0.035 * tenure + 1.3 * (contract == "Month-to-month")
             + 0.5 * (internet == "Fiber optic") + 0.25 * support + 0.012 * (latency - 45)
             + 0.3 * (pay == "Electronic check") + 0.3 * senior)
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"TC-{i:05d}" for i in range(n)],
        "gender": gender, "senior_citizen": senior, "tenure_months": tenure,
        "contract_type": contract, "internet_service": internet, "payment_method": pay,
        "monthly_charges": monthly, "total_charges": np.round(monthly * tenure, 2),
        "monthly_data_gb": np.round(data_gb, 2), "call_minutes": calls, "sms_count": sms,
        "support_calls": support.astype(float), "network_latency_ms": latency, "churn": churn})
    return df

def inject_issues(df):
    df = df.copy()
    n = len(df)
    # missing values
    for col, frac in [("monthly_data_gb", .06), ("support_calls", .04),
                      ("contract_type", .02), ("network_latency_ms", .05)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    # total_charges stored as text with blanks
    df["total_charges"] = df["total_charges"].astype(str)
    df.loc[rng.choice(n, 40, replace=False), "total_charges"] = " "
    # inconsistent categories
    idx = rng.choice(n, 300, replace=False)
    df.loc[idx[:100], "contract_type"] = "month-to-month "
    df.loc[idx[100:180], "contract_type"] = "M2M"
    df.loc[idx[180:240], "contract_type"] = "2 year"
    df.loc[idx[240:], "internet_service"] = "fibre optic"
    df.loc[rng.choice(n, 120, replace=False), "gender"] = "M"
    # outliers / impossible values
    df.loc[rng.choice(n, 25, replace=False), "monthly_data_gb"] = rng.uniform(900, 2500, 25)
    df.loc[rng.choice(n, 10, replace=False), "call_minutes"] = rng.uniform(-50, -1, 10)
    df.loc[rng.choice(n, 15, replace=False), "network_latency_ms"] = rng.uniform(900, 5000, 15)
    # duplicate rows
    df = pd.concat([df, df.sample(100, random_state=1)], ignore_index=True)
    return df.sample(frac=1, random_state=7).reset_index(drop=True)

raw = inject_issues(generate_raw())
raw.to_csv("telecom_raw.csv", index=False)
R["raw_shape"] = list(raw.shape)
R["raw_head"] = raw.head(5).to_dict("records")
R["raw_dtypes"] = raw.dtypes.astype(str).to_dict()
R["missing_before"] = raw.isna().sum()[raw.isna().sum() > 0].to_dict()
R["duplicates_full"] = int(raw.duplicated().sum())

# ---------------------------------------------------------------- 2. CLEANING
df = raw.copy()
# 2.1 duplicates
df = df.drop_duplicates().reset_index(drop=True)
R["after_dedup"] = len(df)

# 2.2 type fixing
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
R["total_charges_blank"] = int(df["total_charges"].isna().sum())

# 2.3 inconsistent categories
R["contract_before"] = df["contract_type"].value_counts(dropna=False).to_dict()
df["contract_type"] = (df["contract_type"].str.strip().str.lower()
    .replace({"m2m": "month-to-month", "2 year": "two year"})
    .map({"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}))
df["internet_service"] = (df["internet_service"].str.strip().str.lower()
    .replace({"fibre optic": "fiber optic"})
    .map({"fiber optic": "Fiber optic", "dsl": "DSL", "no": "No"}))
df["gender"] = df["gender"].replace({"M": "Male"})
R["contract_after"] = df["contract_type"].value_counts(dropna=False).to_dict()
R["internet_after"] = df["internet_service"].value_counts().to_dict()

# 2.4 impossible values -> NaN
neg = int((df["call_minutes"] < 0).sum())
df.loc[df["call_minutes"] < 0, "call_minutes"] = np.nan
R["negative_calls"] = neg

# ---------------------------------------------------------------- 3. OUTLIERS
num_cols = ["monthly_charges", "monthly_data_gb", "call_minutes", "sms_count", "network_latency_ms"]
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)
R["outliers"] = {}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].boxplot(df["monthly_data_gb"].dropna(), vert=False); axes[0].set_title("monthly_data_gb - before")
axes[0].set_xlabel("GB")
before_desc = df[num_cols].describe().round(2)
for c in num_cols:
    lo, hi = iqr_bounds(df[c].dropna(), 3.0 if c == "monthly_data_gb" else 1.5)
    cnt = int(((df[c] < lo) | (df[c] > hi)).sum())
    R["outliers"][c] = {"lower": round(lo, 2), "upper": round(hi, 2), "count": cnt}
    df[c] = df[c].clip(lo, hi)   # winsorise (NaN stay NaN)
axes[1].boxplot(df["monthly_data_gb"].dropna(), vert=False); axes[1].set_title("monthly_data_gb - after capping")
axes[1].set_xlabel("GB")
plt.tight_layout(); plt.savefig("fig_outliers.png", dpi=170); plt.close()
R["desc_after"] = df[num_cols].describe().round(2).to_dict()
R["desc_before"] = before_desc.to_dict()

# ---------------------------------------------------------------- 4. MISSING
R["missing_now"] = df.isna().sum()[df.isna().sum() > 0].to_dict()
# total_charges blanks = brand-new customers -> derive from monthly * tenure
mask = df["total_charges"].isna()
df.loc[mask, "total_charges"] = (df.loc[mask, "monthly_charges"] * df.loc[mask, "tenure_months"]).round(2)
miss = df.isna().mean().mul(100).round(2)
fig, ax = plt.subplots(figsize=(7, 3.2))
m = miss[miss > 0].sort_values()
ax.barh(m.index, m.values, color="#2E75B6"); ax.set_xlabel("% missing (after cleaning, before imputation)")
for i, v in enumerate(m.values): ax.text(v + .05, i, f"{v}%", va="center", fontsize=8)
plt.tight_layout(); plt.savefig("fig_missing.png", dpi=170); plt.close()
R["missing_pct"] = m.to_dict()

# ---------------------------------------------------------------- 5. FEATURE ENGINEERING
df["avg_monthly_spend"] = (df["total_charges"] / df["tenure_months"]).round(2)
df["tenure_group"] = pd.cut(df["tenure_months"], [0, 12, 24, 48, 72],
                            labels=["0-12", "13-24", "25-48", "49-72"])
df["is_heavy_user"] = (df["monthly_data_gb"] > df["monthly_data_gb"].median() * 1.5).astype(float)
df.loc[df["monthly_data_gb"].isna(), "is_heavy_user"] = np.nan

# ---------------------------------------------------------------- 6. PREPROCESSING PIPELINE
y = df["churn"]
X = df.drop(columns=["churn", "customer_id"])
X["tenure_group"] = X["tenure_group"].astype(str)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
R["split"] = {"train": len(Xtr), "test": len(Xte), "churn_rate": round(float(y.mean()), 3)}

numeric = ["tenure_months", "monthly_charges", "total_charges", "monthly_data_gb", "call_minutes",
           "sms_count", "support_calls", "network_latency_ms", "avg_monthly_spend"]
binary = ["senior_citizen", "is_heavy_user"]
categ = ["gender", "contract_type", "internet_service", "payment_method", "tenure_group"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
    ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", drop="first"))]), categ),
])
Xtr_p = pre.fit_transform(Xtr); Xte_p = pre.transform(Xte)
names = pre.get_feature_names_out()
R["n_features_out"] = int(Xtr_p.shape[1])
R["feature_names"] = [n.split("__")[-1] for n in names]
R["scaled_check"] = {n.split("__")[-1]: [round(float(Xtr_p[:, i].mean()), 3), round(float(Xtr_p[:, i].std()), 3)]
                     for i, n in enumerate(names) if n.startswith("num__")}
R["nan_after"] = int(np.isnan(Xtr_p).sum())

# min-max illustration
mm = MinMaxScaler().fit(Xtr[["monthly_data_gb"]].fillna(Xtr["monthly_data_gb"].median()))
R["minmax_demo"] = {"min": float(mm.transform(pd.DataFrame({"monthly_data_gb":[Xtr["monthly_data_gb"].min()]}))[0][0]),
                    "max": float(mm.transform(pd.DataFrame({"monthly_data_gb":[Xtr["monthly_data_gb"].max()]}))[0][0])}

# scaling visual
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].hist(Xtr["monthly_data_gb"].dropna(), bins=40, color="#7F7F7F"); ax[0].set_title("monthly_data_gb (raw scale)")
i = list(pre.get_feature_names_out()).index("num__monthly_data_gb")
ax[1].hist(Xtr_p[:, i], bins=40, color="#2E75B6"); ax[1].set_title("monthly_data_gb (standardised)")
plt.tight_layout(); plt.savefig("fig_scaling.png", dpi=170); plt.close()

# churn by contract chart
fig, ax = plt.subplots(figsize=(6, 3))
cr = df.groupby("contract_type")["churn"].mean().mul(100)
ax.bar(cr.index, cr.values, color=["#2E75B6", "#9DC3E6", "#1F3864"][:len(cr)])
ax.set_ylabel("Churn rate (%)"); ax.set_title("Churn rate by contract type (after cleaning)")
for i, v in enumerate(cr.values): ax.text(i, v + .5, f"{v:.1f}%", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("fig_churn_contract.png", dpi=170); plt.close()
R["churn_by_contract"] = cr.round(1).to_dict()

# ---------------------------------------------------------------- 7. IMPACT ON MODELLING
def evaluate(Xa, Xb):
    m = LogisticRegression(max_iter=1000).fit(Xa, ytr)
    p = m.predict_proba(Xb)[:, 1]
    return {"accuracy": round(accuracy_score(yte, p > .5), 3),
            "f1": round(f1_score(yte, p > .5), 3), "auc": round(roc_auc_score(yte, p), 3)}
R["model_clean"] = evaluate(Xtr_p, Xte_p)

# naive baseline: no cleaning of categories/outliers, drop NaN rows, label-encode
rawb = raw.drop_duplicates().copy()
rawb["total_charges"] = pd.to_numeric(rawb["total_charges"], errors="coerce")
rawb = rawb.dropna()
R["naive_rows_kept"] = len(rawb)
Xb = pd.get_dummies(rawb.drop(columns=["churn", "customer_id"]), drop_first=True).astype(float)
yb = rawb["churn"]
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=.2, stratify=yb, random_state=42)
mb = LogisticRegression(max_iter=20000).fit(Xb_tr, yb_tr)
pb = mb.predict_proba(Xb_te)[:, 1]
R["model_naive"] = {"accuracy": round(accuracy_score(yb_te, pb > .5), 3),
                    "f1": round(f1_score(yb_te, pb > .5), 3), "auc": round(roc_auc_score(yb_te, pb), 3)}
df.to_csv("telecom_clean.csv", index=False)
R["clean_shape"] = list(df.shape)
json.dump(R, open("results.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
