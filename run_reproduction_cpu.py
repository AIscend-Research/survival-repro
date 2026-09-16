"""
CPU smoke-test / validation run of the four review fixes in
VAECox_reproduction.ipynb, executed against the toy pickle data shipped in
this repo (30 patients x 20 cancer types) instead of the real GenoTEX/Xena
data the notebook targets (which needs Kaggle GPU + an attached dataset this
machine doesn't have).

Purpose: prove the four fixes run end-to-end without runtime errors and
produce internally-consistent output, NOT to reproduce the paper's numbers.
Toy-data C-index values are noisy (30 patients/cancer) and not comparable to
either the paper or the real Xena-based reproduction. See REPRODUCIBILITY.md.

Fixes exercised here (ported near-verbatim from the notebook cells):
  1. Per-seed paired statistics (paired_stats) instead of marginal seed std.
  2. Linear-baseline regularization path (PENALTY_GRID) instead of piggy-
     backing CoxLasso/CoxRidge on the neural 3-combo grid.
  3. Robustness sweep over all 5 models + a VAECox-Random ablation, instead
     of CoxRidge-vs-VAECox only.
  4. Leak-free VAE pretraining: eval-cohort test-fold patients (union over
     all seeds) excluded from the pan-cancer pretraining matrix.
"""
import os, sys, pickle, glob, time, copy, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
DEVICE = torch.device("cpu")

CFG = dict(
    PAPER_10=['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG',
              'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD'],
    HIDDEN=512,          # smaller than the paper's 4096 -- CPU/toy-data budget
    LATENT=128,
    VAE_EPOCHS=200,
    VAE_LR=1e-3,
    VAE_WD=1e-5,
    SURV_EPOCHS=100,
    SEEDS=list(range(10)),
    ROBUST_SEEDS=list(range(5)),
    HP_SEARCH=True,
    PRETRAIN_EXCLUDE_TEST=True,
    OUT="results/cpu_smoketest",
)
os.makedirs(CFG["OUT"], exist_ok=True)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------------------
# Data: local toy pickles instead of GenoTEX
# ---------------------------------------------------------------------------
DATA_DIR = "data"
ALL_20 = ['BLCA', 'BRCA', 'CESC', 'COAD', 'GBM', 'HNSC', 'KIRC', 'KIRP',
          'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD',
          'READ', 'SKCM', 'STAD', 'THCA']

COHORT_DFS = {}
for c in ALL_20:
    path = os.path.join(DATA_DIR, f"imputed_and_binary_{c}.pickle")
    df = pickle.load(open(path, "rb"))[0].copy()
    COHORT_DFS[c] = df
GENES = [c for c in COHORT_DFS["BLCA"].columns if c not in ("censored", "survival")]
NUM_FEATURES = len(GENES)
print(f"Loaded {len(COHORT_DFS)} toy cohorts, {NUM_FEATURES} genes")


def cohort_matrix(cohort):
    d = COHORT_DFS[cohort]
    X = d[GENES].values.astype(np.float32)
    y = d["survival"].values.astype(np.float64)
    c = d["censored"].values.astype(np.int32)
    return X, y, c


def make_split(cohort, seed):
    X, y, c = cohort_matrix(cohort)
    try:
        strata = pd.qcut(y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop")
    except Exception:
        strata = c
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(sss.split(X, strata))
    sc = StandardScaler().fit(X[tr])
    return (sc.transform(X[tr]).astype(np.float32), sc.transform(X[te]).astype(np.float32),
            y[tr], y[te], c[tr], c[te])


def split_indices(cohort, seed):
    X, y, c = cohort_matrix(cohort)
    try:
        strata = pd.qcut(y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop")
    except Exception:
        strata = c
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(sss.split(X, strata))


def split_arrays(cohort, tr, te):
    X, y, c = cohort_matrix(cohort)
    sc = StandardScaler().fit(X[tr])
    return (sc.transform(X[tr]).astype(np.float32), sc.transform(X[te]).astype(np.float32),
            y[tr], y[te], c[tr], c[te])


# ---------------------------------------------------------------------------
# Fix 4: leak-free pan-cancer pretraining matrix
# ---------------------------------------------------------------------------
def _held_out_test_rows(cohort, seeds):
    X, y, c = cohort_matrix(cohort)
    try:
        strata = pd.qcut(y, q=min(5, len(np.unique(y))), labels=False, duplicates="drop")
    except Exception:
        strata = c
    held = set()
    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        _, te = next(sss.split(X, strata))
        held.update(te.tolist())
    return held


def pancancer_matrix(exclude_test=None, seeds=None):
    exclude_test = CFG["PRETRAIN_EXCLUDE_TEST"] if exclude_test is None else exclude_test
    seeds = seeds if seeds is not None else CFG["SEEDS"]
    mats, n_dropped = [], {}
    for c in COHORT_DFS:
        X, y, cens = cohort_matrix(c)
        if exclude_test and c in CFG["PAPER_10"]:
            held = _held_out_test_rows(c, seeds)
            keep = np.array([i for i in range(len(X)) if i not in held])
            if len(keep) >= 5:
                n_dropped[c] = len(X) - len(keep)
                X = X[keep]
            else:
                n_dropped[c] = 0
        mats.append(StandardScaler().fit_transform(X).astype(np.float32))
    if exclude_test and n_dropped:
        total = sum(n_dropped.values())
        print(f"pancancer_matrix: excluded {total} eval-cohort test patients "
              f"(union over {len(seeds)} seeds) from VAE pretraining")
    return np.vstack(mats)


# ---------------------------------------------------------------------------
# Models (ported from the notebook)
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, num_features, hidden=4096, latent=128, dropout=0.0):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(num_features, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.encode_mu = nn.Sequential(nn.Linear(hidden, latent), nn.Tanh(), nn.Dropout(dropout))
        self.encode_si = nn.Sequential(nn.Linear(hidden, latent), nn.Tanh(), nn.Dropout(dropout))
        self.decode = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Dropout(dropout),
                                    nn.Linear(hidden, num_features))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        h = self.encode(x)
        mu, logvar = self.encode_mu(h), self.encode_si(h)
        recon = self.decode(mu)
        mse = F.mse_loss(recon, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld


class PartialNLL(nn.Module):
    def forward(self, theta, R, censored):
        observed = 1 - censored
        num_obs = torch.sum(observed)
        if num_obs == 0:
            return (theta * 0).sum()
        exp_theta = torch.exp(theta)
        return -(torch.sum((theta.reshape(-1) -
                 torch.log(torch.sum(exp_theta * R.t(), 0))) * observed) / num_obs)


class CoxLinear(nn.Module):
    def __init__(self, p):
        super().__init__(); self.fc1 = nn.Linear(p, 1); nn.init.xavier_normal_(self.fc1.weight)
    def forward(self, x): return self.fc1(x)


class Coxnnet(nn.Module):
    def __init__(self, p):
        super().__init__(); h = int(np.ceil(p ** 0.5))
        self.fc1 = nn.Linear(p, h); self.fc2 = nn.Linear(h, 1)
    def forward(self, x): return self.fc2(torch.tanh(self.fc1(x)))


class CoxMLP(nn.Module):
    def __init__(self, p, nhid=100, dropout=0.0):
        super().__init__(); self.fc1 = nn.Linear(p, nhid); self.fc2 = nn.Linear(nhid, 1); self.d = dropout
    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), self.d, training=self.training); return self.fc2(x)


class VAECox(nn.Module):
    def __init__(self, pretrained_vae, latent=128):
        super().__init__()
        vae = copy.deepcopy(pretrained_vae)
        self.encode = vae.encode
        self.encode_mu = vae.encode_mu
        self.cox = Coxnnet(latent)
        for p in self.parameters():
            p.requires_grad = True
    def forward(self, x):
        return self.cox(self.encode_mu(self.encode(x)))


def make_R(y):
    n = len(y); R = np.zeros((n, n), dtype=np.float32)
    for i in range(n): R[i, :] = (y >= y[i])
    return R


def cindex_safe(y, pred, c):
    ev = (c == 0)
    if ev.sum() == 0: return float("nan")
    try: return concordance_index(y, pred, ev)
    except Exception: return float("nan")


def train_eval(model, Xtr, ytr, ctr, Xte, yte, cte, lr, wd, epochs, lasso=0.0):
    model = model.to(DEVICE)
    lossf = PartialNLL()
    X = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    R = torch.tensor(make_R(ytr), dtype=torch.float32, device=DEVICE)
    c = torch.tensor(ctr, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(); theta = model(X); loss = lossf(theta, R, c)
        if lasso > 0:
            loss = loss + lasso * sum(p.abs().sum() for p in model.fc1.parameters())
        if torch.isnan(loss) or torch.isinf(loss): break
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = -model(torch.tensor(Xte, dtype=torch.float32, device=DEVICE)).reshape(-1).cpu().numpy()
    return cindex_safe(yte, pred, cte)


def _fit(model, Xtr, ytr, ctr, lr, wd, epochs, lasso=0.0):
    model = model.to(DEVICE)
    lossf = PartialNLL()
    X = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    R = torch.tensor(make_R(ytr), dtype=torch.float32, device=DEVICE)
    c = torch.tensor(ctr, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(model(X), R, c)
        if lasso > 0:
            loss = loss + lasso * sum(p.abs().sum() for p in model.fc1.parameters())
        if torch.isnan(loss) or torch.isinf(loss): break
        loss.backward(); opt.step()
    model.eval()
    return model


def _risk(model, X):
    with torch.no_grad():
        return -model(torch.tensor(X, dtype=torch.float32, device=DEVICE)).reshape(-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Fix 2: linear-baseline regularization path, kept separate from the neural grid
# ---------------------------------------------------------------------------
NEURAL_HP_GRID = [(1e-3, 1e-5), (1e-3, 1e-3), (1e-4, 1e-5)] if CFG["HP_SEARCH"] else [(1e-3, 1e-5)]
LINEAR_LR = 1e-4
PENALTY_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0] \
    if CFG["HP_SEARCH"] else [1e-3]

MODELS = ["CoxLasso", "CoxRidge", "Coxnnet", "CoxMLP", "VAECox"]
DEFAULT_HP = {"CoxLasso": (1e-4, 0.0), "CoxRidge": (1e-4, 1e-3), "Coxnnet": (1e-3, 1e-5),
              "CoxMLP": (1e-3, 1e-5), "VAECox": (1e-3, 1e-5), "VAECox-Random": (1e-3, 1e-5)}


def build_model(name, p, vae_model):
    if name == "CoxLasso":  return CoxLinear(p)
    if name == "CoxRidge":  return CoxLinear(p)
    if name == "Coxnnet":   return Coxnnet(p)
    if name == "CoxMLP":    return CoxMLP(p)
    if name == "VAECox":    return VAECox(vae_model, CFG["LATENT"])
    if name == "VAECox-Random": return VAECox(VAE(NUM_FEATURES, CFG["HIDDEN"], CFG["LATENT"]), CFG["LATENT"])
    raise ValueError(name)


def fit_one(name, Xtr, ytr, ctr, Xte, yte, cte, hp, vae_model):
    m = build_model(name, Xtr.shape[1], vae_model)
    if name == "CoxLasso":
        return train_eval(m, Xtr, ytr, ctr, Xte, yte, cte, LINEAR_LR, 0.0, CFG["SURV_EPOCHS"], lasso=hp)
    if name == "CoxRidge":
        return train_eval(m, Xtr, ytr, ctr, Xte, yte, cte, LINEAR_LR, hp, CFG["SURV_EPOCHS"])
    lr, wd = hp
    return train_eval(m, Xtr, ytr, ctr, Xte, yte, cte, lr, wd, CFG["SURV_EPOCHS"])


def search_hp(name, cohort, vae_model):
    grid = PENALTY_GRID if name in ("CoxLasso", "CoxRidge") else NEURAL_HP_GRID
    if len(grid) == 1:
        return grid[0]
    Xtr, Xte, ytr, yte, ctr, cte = make_split(cohort, 0)
    best, best_hp = -1, grid[0]
    for hp in grid:
        ci = fit_one(name, Xtr, ytr, ctr, Xte, yte, cte, hp, vae_model)
        if not np.isnan(ci) and ci > best:
            best, best_hp = ci, hp
    return best_hp


def fit_risk(name, Xtr, ytr, ctr, Xte, vae_model, lr=None, wd=None, epochs=None):
    lr0, wd0 = DEFAULT_HP[name]
    lr = lr0 if lr is None else lr
    wd = wd0 if wd is None else wd
    epochs = epochs or CFG["SURV_EPOCHS"]
    m = build_model(name, Xtr.shape[1], vae_model)
    m = _fit(m, Xtr, ytr, ctr, lr, wd, epochs, 0.01 if name == "CoxLasso" else 0.0)
    return m, _risk(m, Xte)


# ---------------------------------------------------------------------------
# Phase 2 + Fix 1: per-seed paired stats
# ---------------------------------------------------------------------------
def run_phase2(vae_model, cohorts=None, seeds=None):
    cohorts = cohorts or CFG["PAPER_10"]
    seeds = seeds if seeds is not None else CFG["SEEDS"]
    rows, seed_rows = [], []
    for cohort in cohorts:
        print(f"\n-- {cohort} --")
        hp = {m: search_hp(m, cohort, vae_model) for m in MODELS}
        for m in MODELS:
            vals = []
            for seed in seeds:
                set_seed(seed)
                Xtr, Xte, ytr, yte, ctr, cte = make_split(cohort, seed)
                ci = fit_one(m, Xtr, ytr, ctr, Xte, yte, cte, hp[m], vae_model)
                vals.append(ci)
                seed_rows.append(dict(cancer=cohort, model=m, seed=seed, cindex=ci))
            v = [x for x in vals if not np.isnan(x)]
            mean = np.mean(v) if v else float("nan")
            std = np.std(v) if v else float("nan")
            rows.append(dict(cancer=cohort, model=m, mean_cindex=round(mean, 4),
                             std_cindex=round(std, 4), n_valid=len(v)))
            print(f"  {m:9s}: {mean:.3f} +/- {std:.3f}  (hp={hp[m]}, {len(v)} seeds)")
    df = pd.DataFrame(rows)
    seed_df = pd.DataFrame(seed_rows)
    df.to_csv(f"{CFG['OUT']}/cindex_long.csv", index=False)
    seed_df.to_csv(f"{CFG['OUT']}/cindex_by_seed.csv", index=False)
    wide = df.pivot(index="model", columns="cancer", values="mean_cindex")
    wide["Mean"] = wide.mean(axis=1)
    wide.to_csv(f"{CFG['OUT']}/cindex_comparison.csv")
    wins = {m: 0 for m in MODELS}
    for cohort in wide.columns[:-1]:
        col = wide[cohort].dropna()
        if len(col): wins[col.idxmax()] += 1
    print("\n=== WINS ===")
    for m, w in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {m:9s}: {w}/{len(cohorts)}")
    return df, seed_df, wide, wins


def paired_stats(seed_df, baseline_models=("CoxLasso", "CoxRidge", "Coxnnet", "CoxMLP")):
    if seed_df is None or not len(seed_df):
        print("paired_stats: no per-seed data -- skipping")
        return pd.DataFrame()
    rows = []
    wide = seed_df.pivot_table(index=["cancer", "seed"], columns="model", values="cindex")
    for cohort, g in wide.groupby(level=0):
        g = g.droplevel(0)
        if "VAECox" not in g.columns:
            continue
        for base in baseline_models:
            if base not in g.columns:
                continue
            paired = g[["VAECox", base]].dropna()
            if len(paired) < 3:
                continue
            diff = (paired["VAECox"] - paired[base]).values
            n = len(diff)
            wins = int((diff > 0).sum())
            rng = np.random.default_rng(42)
            boot = np.array([rng.choice(diff, n, replace=True).mean() for _ in range(2000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            try:
                _, p = wilcoxon(diff) if np.any(diff != 0) else (np.nan, 1.0)
            except ValueError:
                p = 1.0
            rows.append(dict(cancer=cohort, baseline=base, n_seeds=n,
                             mean_diff=round(float(np.mean(diff)), 4),
                             wins_vaecox=wins, losses=n - wins,
                             ci95_lo=round(float(lo), 4), ci95_hi=round(float(hi), 4),
                             wilcoxon_p=round(float(p), 4)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{CFG['OUT']}/paired_stats.csv", index=False)
    if len(df):
        sig = df[df.wilcoxon_p < 0.05]
        print(f"paired_stats: {len(df)} (cancer, baseline) pairs; "
              f"{len(sig)} significant at p<0.05 (Wilcoxon signed-rank), "
              f"{int((df.mean_diff > 0).sum())} with VAECox ahead on average")
    return df


# ---------------------------------------------------------------------------
# Fix 3: robustness sweep over all models + VAECox-Random ablation
# ---------------------------------------------------------------------------
ROBUSTNESS_MODELS = ("CoxLasso", "CoxRidge", "Coxnnet", "CoxMLP", "VAECox", "VAECox-Random")


def robustness_all(vae_model, cohorts=None, seeds=None, models=ROBUSTNESS_MODELS,
                   miss=(0.0, 0.1, 0.25, 0.5), sigmas=(0.0, 0.5, 1.0, 2.0)):
    cohorts = cohorts or CFG["PAPER_10"]
    seeds = seeds if seeds is not None else CFG["ROBUST_SEEDS"]
    rows = []
    for cohort in cohorts:
        acc = {}
        for seed in seeds:
            set_seed(seed)
            tr, te = split_indices(cohort, seed)
            Xtr, Xte, ytr, yte, ctr, cte = split_arrays(cohort, tr, te)
            fitted = {m: fit_risk(m, Xtr, ytr, ctr, Xte, vae_model)[0] for m in models}
            rng_m = np.random.default_rng(seed + 1000)
            rng_n = np.random.default_rng(seed + 2000)
            for frac in miss:
                Xc = Xte * (rng_m.random(Xte.shape) >= frac)
                for m, mod in fitted.items():
                    acc.setdefault(("missing", f"{int(frac*100)}%"), {}).setdefault(m, []).append(
                        cindex_safe(yte, _risk(mod, Xc.astype(np.float32)), cte))
            for sig in sigmas:
                Xc = (Xte + rng_n.normal(0, sig, Xte.shape)).astype(np.float32)
                for m, mod in fitted.items():
                    acc.setdefault(("noise", f"sigma={sig}"), {}).setdefault(m, []).append(
                        cindex_safe(yte, _risk(mod, Xc), cte))
        for (exp, lev), d in acc.items():
            row = dict(cancer=cohort, experiment=exp, level=lev)
            for m in models:
                row[m] = round(float(np.nanmean(d.get(m, [np.nan]))), 4)
            rows.append(row)
        print(f"  {cohort} done")
    df = pd.DataFrame(rows)
    df.to_csv(f"{CFG['OUT']}/robustness_by_cancer.csv", index=False)
    for exp in ("missing", "noise"):
        sub = df[df.experiment == exp]
        if not len(sub): continue
        piv = sub.pivot_table(index="level", values=list(models), aggfunc="mean")
        clean = piv.iloc[0]
        rel = (piv - clean) / clean * 100
        print(f"\n  {exp}: % change in C-index vs clean (mean over cohorts)")
        print(rel.round(1).to_string())
    if "VAECox" in df.columns and "VAECox-Random" in df.columns:
        for exp, lev in [("missing", f"{int(max(miss)*100)}%"), ("noise", f"sigma={max(sigmas)}")]:
            sub = df[(df.experiment == exp) & (df.level == lev)]
            if len(sub):
                delta = (sub["VAECox"] - sub["VAECox-Random"]).mean()
                print(f"  pretraining effect @ {exp}={lev}: "
                      f"VAECox - VAECox-Random = {delta:+.4f} C-index (mean over cohorts)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("CPU smoke-test of review fixes -- toy data, NOT paper-scale numbers")
    print("=" * 70)

    X_PAN = pancancer_matrix()
    print(f"Pan-cancer VAE matrix: {X_PAN.shape}  "
          f"[test-leakage {'excluded' if CFG['PRETRAIN_EXCLUDE_TEST'] else 'NOT excluded'}]")

    set_seed(0)
    vae = VAE(NUM_FEATURES, CFG["HIDDEN"], CFG["LATENT"]).to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=CFG["VAE_LR"], weight_decay=CFG["VAE_WD"])
    Xp = torch.tensor(X_PAN, dtype=torch.float32)
    tvae0 = time.time()
    for ep in range(CFG["VAE_EPOCHS"]):
        vae.train(); opt.zero_grad(); loss = vae(Xp); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == CFG["VAE_EPOCHS"] - 1:
            print(f"  VAE epoch {ep:4d}  loss {loss.item():.4f}")
    vae.eval()
    print(f"VAE pretrained in {time.time()-tvae0:.1f}s")
    torch.save(vae.state_dict(), f"{CFG['OUT']}/vae_pretrained_leakfree.pt")

    print("\n" + "=" * 70)
    print("PHASE 2 (with linear regularization path, per-seed saving)")
    print("=" * 70)
    PH2_LONG, PH2_SEED, PH2_WIDE, PH2_WINS = run_phase2(vae)
    print("\n", PH2_WIDE.round(3))

    print("\n" + "=" * 70)
    print("PAIRED STATISTICS (fix for the marginal-std argument)")
    print("=" * 70)
    PAIRED = paired_stats(PH2_SEED)
    if len(PAIRED):
        print(PAIRED.to_string(index=False))

    print("\n" + "=" * 70)
    print("ROBUSTNESS SWEEP (all models + VAECox-Random ablation)")
    print("=" * 70)
    ROB = robustness_all(vae)

    print(f"\nDone in {time.time()-t0:.1f}s. Outputs in {CFG['OUT']}/")
