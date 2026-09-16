# results/tcga update — review-fix re-run

Files updated by this commit (Phase 2 + robustness only, real GenoTEX/Xena data,
all 10 paper cancers): `cindex_comparison.csv`, `cindex_long.csv`,
`robustness_by_cancer.csv`, and their figures. Newly added:
`cindex_by_seed.csv`, `paired_stats.csv`.

These replace pre-review-fix numbers. In particular, the old
`robustness_by_cancer.csv` only scored CoxRidge vs VAECox; the new one scores
all 5 models plus a VAECox-Random ablation (identical architecture, untrained
encoder), per the reviewer feedback addressed in PR #2.

**Config used for this run** (trimmed to fit a time budget, not the paper's
full settings):
- VAE pretrained for 100 epochs (paper/notebook default: 500)
- Phase 2: all 10 seeds, full linear-penalty-path + neural-grid search (unchanged from default)
- Robustness: default 5 seeds, all 10 cancers, 6 models
- VAE pretraining pool excludes eval-cohort test patients (union over all 10 seeds) — `CFG["PRETRAIN_EXCLUDE_TEST"]=True`

**Not updated by this commit** — still reflect the pre-fix run, untouched
because this run didn't recompute them (they aren't affected by the four
review fixes): `fairness.csv`, `cohort_fairness.csv`, `feature_importance.csv`,
`feature_subset.csv`, `km_summary.csv`, `lightweight_by_cancer.csv`,
`lightweight_disparity.csv`, `permutation_importance.csv`, `subgroup_*.csv`,
`manuscript_numbers.json`, `reproducibility_card.txt`, and their figures.

**Headline findings from this run:**
- VAECox wins 5/10 cancers (BLCA, LIHC, LUAD, OV, STAD); mean C-index
  essentially tied with Coxnnet (0.6459 vs 0.6461).
- Paired per-seed stats: 8/40 (cancer, baseline) comparisons significant at
  p<0.05 — 6 favor VAECox, 2 favor a baseline (HNSC vs CoxRidge, KIRC vs
  Coxnnet). Not "all within noise," but not uniformly favoring VAECox either.
- Robustness ablation: VAECox's average advantage over VAECox-Random across
  all 10 cancers is ~0 under both missing-feature and noise corruption —
  no measurable robustness benefit from pretraining once architecture is
  held constant.

A full-scale re-run (500 VAE epochs, 10 robustness seeds, plus the skipped
extensions) would give more complete numbers but wasn't done here due to
time constraints.
