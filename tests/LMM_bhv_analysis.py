"""
Linear Mixed Models for Fixed-Interval Behavioral Task
=======================================================
Three model families:
  1. P(TP)       — logistic LMM: probability of a trial having a transition point
  2. TP value    — Gaussian LMM: transition point timing (absolute seconds + normalized)
  3. Pressing rate — Gaussian LMM: post-TP lever press rate

Data hierarchy: trials → blocks → sessions → animals
Between-subject factor: cue_condition (cued / uncued)
Within-subject factors: fi (15/30/60 s), reward_magnitude (7/14/28 nprotocols),
                        exp_condition (a/b/c), block_position (block index in session)

Author: generated for PhD/postdoc FI project
Dependencies: pandas, numpy, statsmodels, scipy, matplotlib
"""
#%%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 0. EXPECTED DATA STRUCTURE
# =============================================================================
# Your DataFrame should have one row per trial with (at minimum) these columns:
#
#   animal_id        : str/int   – unique animal identifier
#   session_id       : str/int   – unique session identifier (globally unique, or
#                                   use animal_id + session number together)
#   block_id         : str/int   – unique block identifier (globally unique)
#   block_position   : int       – ordinal position of block within session (1-based)
#   fi               : float     – fixed interval in seconds (15, 30, or 60)
#   reward_magnitude : float     – reward magnitude (7, 14, or 28 nprotocols)
#   exp_condition    : str       – 'a', 'b', or 'c'
#   cue_condition    : str       – 'cued' or 'uncued'  (between-subject)
#   has_tp           : int       – 1 if trial has a transition point, 0 otherwise
#   tp_seconds       : float     – transition point in seconds (NaN if no TP)
#   tp_norm          : float     – transition point normalised to [0,1] of FI
#                                   i.e. tp_seconds / fi  (NaN if no TP)
#   pressing_rate    : float     – post-TP press rate in Hz (NaN if no TP)


def load_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic validation and preprocessing.
    Call this before fitting any model.
    """
    required = [
        "animal_id", "session_id", "block_id", "block_position",
        "fi", "reward_magnitude", "exp_condition", "cue_condition",
        "has_tp", "tp_seconds", "tp_norm", "pressing_rate",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # --- Categorical coding (treatment coding; baselines chosen deliberately) ---
    # FI: 15 s as baseline (smallest interval → reference for "easy timing")
    df["fi_cat"] = pd.Categorical(df["fi"].astype(str),
                                   categories=["15", "30", "60"])
    # Reward magnitude: 14 as baseline (mid-level; condition 'a' fixed value)
    df["rwd_cat"] = pd.Categorical(df["reward_magnitude"].astype(str),
                                    categories=["7", "14", "28"])
    # Experimental condition: 'a' as baseline (FI-varying, fixed reward)
    df["exp_cond"] = pd.Categorical(df["exp_condition"],
                                     categories=["a", "b", "c"])
    # Cue condition: 'uncued' as baseline (more general group)
    df["cue_cond"] = pd.Categorical(df["cue_condition"],
                                     categories=["uncued", "cued"])

    # Centre block_position to reduce collinearity with intercept
    df["block_pos_c"] = df["block_position"] - df["block_position"].mean()

    # Log-transform pressing rate (typically right-skewed, strictly positive)
    df["log_pressing_rate"] = np.log(df["pressing_rate"])

    print("=== Data summary ===")
    print(f"  Trials total       : {len(df)}")
    print(f"  Animals            : {df['animal_id'].nunique()}")
    print(f"  Sessions           : {df['session_id'].nunique()}")
    print(f"  Blocks             : {df['block_id'].nunique()}")
    print(f"  Trials with TP     : {df['has_tp'].sum()} "
          f"({100*df['has_tp'].mean():.1f}%)")
    print(f"  Cue groups         : {df['cue_condition'].value_counts().to_dict()}")
    print(f"  Exp conditions     : {df['exp_condition'].value_counts().to_dict()}")
    print()
    return df


# =============================================================================
# 1. RANDOM-EFFECTS STRUCTURE (shared across all models)
# =============================================================================
# Trials are nested in blocks, nested in sessions, nested in animals.
# We use a maximal-but-feasible random intercept structure.
# Random slopes are omitted at the innermost levels to avoid singular fits
# with typical rodent-neuroscience sample sizes (n ≈ 6-12 animals).
#
# Formula fragment:  (1 | animal_id/session_id/block_id)
# This expands to:   (1 | animal_id) + (1 | animal_id:session_id)
#                                     + (1 | animal_id:session_id:block_id)
#
# If you have globally unique session_id and block_id labels you can also write:
#   (1 | animal_id) + (1 | session_id) + (1 | block_id)
# Both are equivalent when IDs are nested and unique.

RANDOM = "animal_id / session_id / block_id"   # statsmodels groups= syntax


# =============================================================================
# 2. FIXED-EFFECTS FORMULAS
# =============================================================================
# Interactions selected on theoretical grounds:
#   fi_cat × exp_cond        — does FI effect on timing depend on condition?
#   rwd_cat × exp_cond       — does reward effect on vigor depend on condition?
#   fi_cat × rwd_cat         — direct FI × reward interaction
#   block_pos_c × fi_cat     — do animals adapt speed to FI across blocks?
#   cue_cond                 — main effect of cue (between-subject)
#
# Note: cue_cond × fi_cat and cue_cond × rwd_cat could be added if you have
# a specific hypothesis about whether cuing differentially affects short vs.
# long FIs or different reward levels. Add them with the helpers below.

FIXED_TP = (
    "fi_cat * exp_cond"
    " + rwd_cat * exp_cond"
    " + fi_cat * rwd_cat"
    " + block_pos_c * fi_cat"
    " + cue_cond"
)

FIXED_RATE = (
    "fi_cat * exp_cond"
    " + rwd_cat * exp_cond"
    " + fi_cat * rwd_cat"
    " + block_pos_c * fi_cat"
    " + cue_cond"
)


# =============================================================================
# 3. MODEL 1 — P(transition point exists)
# =============================================================================
# BinomialBayesMixedGLM from statsmodels is brittle when there are many random-
# effect levels (hundreds of session/block dummies) and raises a zero-size array
# error. We therefore use two practical alternatives:
#
#   Option A (default) — Linear Probability LMM via MixedLM
#     Fits has_tp as a continuous outcome with a Gaussian LMM.
#     Coefficients are interpreted as percentage-point changes in P(TP).
#     Works well when P(TP) is not near 0 or 1, which is typical here.
#     Uses the same vc_formula nesting as Models 2 & 3 — fully consistent.
#
#   Option B — Logistic GEE (population-averaged)
#     Uses Generalized Estimating Equations with an exchangeable working
#     correlation within animals. Gives true log-odds coefficients.
#     Less granular than a full GLMM but robust and widely accepted.
#
#   Option C — pymer4 (if R + lme4 installed)
#     Uncomment the pymer4 block to get a proper logistic GLMM via lme4.
#     Install: pip install pymer4  (requires R with lme4 package).

def fit_tp_probability_model(df: pd.DataFrame, method: str = "lpm"):
    """
    Model the probability of a trial having a transition point.

    Parameters
    ----------
    df     : full trial-level DataFrame (all trials, not just TP subset)
    method : 'lpm' (default) — linear probability LMM via MixedLM
             'gee'           — logistic GEE clustered on animal_id
             'pymer4'        — logistic GLMM via R/lme4 (requires pymer4)

    Returns
    -------
    Fitted model result object.
    """
    print("=" * 60)
    print(f"MODEL 1: P(transition point) — [{method.upper()}]")
    print("=" * 60)
    print(f"  N trials: {len(df)}  |  P(TP) overall: {df['has_tp'].mean():.3f}")

    # ------------------------------------------------------------------
    # Option A: Linear Probability Model (LPM) — Gaussian LMM
    # ------------------------------------------------------------------
    if method == "lpm":
        print("\n  [Linear Probability Model — coefficients = Δ P(TP)]\n")

        vc = {
            "session": "0 + C(session_id)",
            "block":   "0 + C(block_id)",
        }
        model = smf.mixedlm(
            formula=f"has_tp ~ {FIXED_TP}",
            data=df,
            groups=df["animal_id"],
            vc_formula=vc,
        )
        result = model.fit(reml=True, method="lbfgs")
        print(result.summary())

        # Sanity check: flag if fitted values are outside [0, 1]
        out_of_range = ((result.fittedvalues < 0) |
                        (result.fittedvalues > 1)).mean()
        if out_of_range > 0.05:
            print(f"\n  ⚠  {100*out_of_range:.1f}% of fitted values outside [0,1].")
            print("     Consider switching to method='gee' for logit-scale results.")
        return result

    # ------------------------------------------------------------------
    # Option B: Logistic GEE — population-averaged log-odds
    # ------------------------------------------------------------------
    elif method == "gee":
        print("\n  [Logistic GEE — coefficients = log-odds, clustered on animal]\n")
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable

        # ----------------------------------------------------------------
        # The zero-size array error occurs because statsmodels' formula
        # parser struggles with Categorical dtype columns in interaction
        # terms — it produces empty dummy columns for level combinations
        # that don't exist in the data, then fails on reduction ops.
        #
        # Fix: build the design matrix explicitly with pd.get_dummies(),
        # drop empty columns, then pass X/y arrays directly to GEE().
        # ----------------------------------------------------------------
        df = df.copy()

        # Dummies with explicit baselines (treatment coding)
        # Baselines: fi=15, rwd=14, exp_cond=a, cue=uncued
        fi_dummies  = pd.get_dummies(df["fi"].astype(str),
                                     prefix="fi",  dtype=float
                                     ).drop(columns=["fi_15.0"], errors="ignore")
        rwd_dummies = pd.get_dummies(df["reward_magnitude"].astype(str),
                                     prefix="rwd", dtype=float
                                     ).drop(columns=["rwd_14.0"], errors="ignore")
        exp_dummies = pd.get_dummies(df["exp_condition"],
                                     prefix="exp", dtype=float
                                     ).drop(columns=["exp_a"], errors="ignore")
        cue_dummy   = (df["cue_condition"] == "cued").astype(float).rename("cued")
        block_pos_c = df["block_pos_c"].astype(float)

        X = pd.concat([fi_dummies, rwd_dummies, exp_dummies,
                        cue_dummy, block_pos_c], axis=1)

        # Build interactions explicitly — avoids empty column problem
        for fc in fi_dummies.columns:
            for ec in exp_dummies.columns:
                X[f"{fc}:{ec}"] = fi_dummies[fc] * exp_dummies[ec]
        for rc in rwd_dummies.columns:
            for ec in exp_dummies.columns:
                X[f"{rc}:{ec}"] = rwd_dummies[rc] * exp_dummies[ec]
        for fc in fi_dummies.columns:
            for rc in rwd_dummies.columns:
                X[f"{fc}:{rc}"] = fi_dummies[fc] * rwd_dummies[rc]
        for fc in fi_dummies.columns:
            X[f"block_pos_c:{fc}"] = block_pos_c * fi_dummies[fc]

        # Drop all-zero columns (combinations absent from data)
        zero_cols = X.columns[X.sum() == 0].tolist()
        if zero_cols:
            print(f"  Dropping {len(zero_cols)} empty columns: {zero_cols}")
            X = X.drop(columns=zero_cols)

        X.insert(0, "Intercept", 1.0)
        col_names = X.columns.tolist()
        X = X.values.astype(float)
        y = df["has_tp"].astype(float).values
        groups = df["animal_id"].values

        model = GEE(
            endog=y,
            exog=X,
            groups=groups,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        result = model.fit()

        # Readable output with named params
        params  = pd.Series(result.params,  index=col_names)
        pvals   = pd.Series(result.pvalues, index=col_names)
        ci      = pd.DataFrame(result.conf_int(), index=col_names,
                                columns=["CI_low", "CI_high"])
        sig     = pvals.apply(lambda p: "***" if p<.001 else
                                         "**"  if p<.01  else
                                         "*"   if p<.05  else "")

        coef_table = pd.concat([params.rename("coef"), ci, pvals.rename("p"), sig.rename("sig")], axis=1)
        print("\n  Coefficients (log-odds):")
        print(coef_table.round(4).to_string())

        or_table = coef_table.copy()
        or_table[["coef","CI_low","CI_high"]] = np.exp(coef_table[["coef","CI_low","CI_high"]])
        or_table = or_table.rename(columns={"coef":"OR"})
        print("\n  Odds Ratios:")
        print(or_table.round(4).to_string())
        return result

    # ------------------------------------------------------------------
    # Option C: pymer4 — proper logistic GLMM via R/lme4
    # ------------------------------------------------------------------
    elif method == "pymer4":
        print("\n  [Logistic GLMM via pymer4/lme4]\n")
        try:
            from pymer4.models import Lmer
        except ImportError:
            raise ImportError(
                "pymer4 is not installed. Run: pip install pymer4\n"
                "Also requires R with lme4: install.packages('lme4')"
            )

        # lme4 formula with nested random intercepts
        lme4_re = "(1 | animal_id / session_id / block_id)"
        lme4_formula = f"has_tp ~ {FIXED_TP} + {lme4_re}"

        model = Lmer(lme4_formula, data=df, family="binomial")
        result = model.fit()
        print(result)
        return model

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'lpm', 'gee', or 'pymer4'.")


# =============================================================================
# 4. MODEL 2 — Gaussian LMM: Transition point value
#    Run twice: absolute seconds and normalized (0–1)
# =============================================================================

def fit_tp_value_model(df: pd.DataFrame, outcome: str = "tp_seconds"):
    """
    Outcome : tp_seconds  OR  tp_norm
    Family  : Gaussian (identity link)
    Subset  : only trials with a TP (has_tp == 1)
    
    outcome='tp_seconds' : model in absolute time; FI in formula as covariate
                           captures scale differences across intervals
    outcome='tp_norm'    : model in proportion of FI; FI effect now reflects
                           Weber-law-like deviations from proportional timing
    
    Key expected results:
      - fi_cat should be a strong positive predictor for tp_seconds
      - fi_cat should NOT be significant for tp_norm (scalar timing)
      - cue_cond should not be significant (cuing affects vigor, not timing)
      - rwd_cat should not be significant (reward affects vigor, not timing)
    """
    label = "absolute (s)" if outcome == "tp_seconds" else "normalised (prop. FI)"
    print("=" * 60)
    print(f"MODEL 2: Transition point — Gaussian LMM [{label}]")
    print("=" * 60)

    df_tp = df[df["has_tp"] == 1].copy()
    print(f"  N trials (TP subset): {len(df_tp)}")

    formula = f"{outcome} ~ {FIXED_TP}"

    # Variance components for nested random effects
    vc = {
        "session": "0 + C(session_id)",
        "block":   "0 + C(block_id)",
    }

    model = smf.mixedlm(
        formula=formula,
        data=df_tp,
        groups=df_tp["animal_id"],
        vc_formula=vc,
    )
    result = model.fit(reml=True, method="lbfgs")

    print(result.summary())
    _print_effect_sizes(result, df_tp, outcome)
    return result


def compare_tp_parameterizations(df: pd.DataFrame):
    """
    Fit both TP models and compare AIC/BIC.
    Also reports whether fi_cat is significant in each — the key test
    of scalar timing: if animals time proportionally, fi effect vanishes
    in the normalised model.
    """
    res_abs  = fit_tp_value_model(df, outcome="tp_seconds")
    res_norm = fit_tp_value_model(df, outcome="tp_norm")

    print("\n=== TP parameterization comparison ===")
    print(f"  tp_seconds model  — AIC: {res_abs.aic:.1f}  BIC: {res_abs.bic:.1f}")
    print(f"  tp_norm model     — AIC: {res_norm.aic:.1f}  BIC: {res_norm.bic:.1f}")
    print()
    print("  Interpretation guide:")
    print("  • If fi_cat is NS in tp_norm but significant in tp_seconds →")
    print("    animals show scalar timing (TP scales proportionally with FI).")
    print("  • If fi_cat remains significant in tp_norm → supra/sub-proportional")
    print("    timing shift (e.g., animals start proportionally earlier for long FIs).")
    return res_abs, res_norm


# =============================================================================
# 5. MODEL 3 — Gaussian LMM: Pressing rate (vigor)
# =============================================================================

def fit_pressing_rate_model(df: pd.DataFrame):
    """
    Outcome : log(pressing_rate)  — log-transform for normality & homoscedasticity
    Family  : Gaussian
    Subset  : only trials with a TP (has_tp == 1)
    
    Key expected results:
      - rwd_cat should be a significant predictor (reward drives vigor)
      - fi_cat should NOT be significant (vigor independent of timing demands)
      - cue_cond should be significant and negative (cuing reduces pressing rate)
      - exp_cond 'c' (constant FI, varying reward) should show clearest rwd effect
    """
    print("=" * 60)
    print("MODEL 3: Pressing rate (vigor) — Gaussian LMM [log scale]")
    print("=" * 60)

    df_tp = df[df["has_tp"] == 1].copy()
    print(f"  N trials (TP subset): {len(df_tp)}")

    formula = f"log_pressing_rate ~ {FIXED_RATE}"

    vc = {
        "session": "0 + C(session_id)",
        "block":   "0 + C(block_id)",
    }

    model = smf.mixedlm(
        formula=formula,
        data=df_tp,
        groups=df_tp["animal_id"],
        vc_formula=vc,
    )
    result = model.fit(reml=True, method="lbfgs")

    print(result.summary())
    _print_effect_sizes(result, df_tp, "log_pressing_rate")
    return result


# =============================================================================
# 6. HELPER: Effect sizes and post-hoc comparisons
# =============================================================================

def _print_effect_sizes(result, df_sub: pd.DataFrame, outcome: str):
    """
    Print marginal and conditional R² (Nakagawa & Schielzeth 2013 approximation)
    for Gaussian LMMs.
    """
    # Variance of fixed effects (marginal)
    fitted_fe = result.fittedvalues
    var_fe = np.var(fitted_fe)

    # Residual variance
    var_resid = result.scale

    # Random-effect variances
    var_re = sum(result.cov_re.values.flatten()) if hasattr(result, "cov_re") else 0.0

    r2_marginal    = var_fe / (var_fe + var_re + var_resid)
    r2_conditional = (var_fe + var_re) / (var_fe + var_re + var_resid)

    print(f"\n  Approx. R² (Nakagawa & Schielzeth):")
    print(f"    Marginal  (fixed effects only): {r2_marginal:.3f}")
    print(f"    Conditional (fixed + random)  : {r2_conditional:.3f}")


def post_hoc_pairwise(result, term: str, levels: list, df_sub: pd.DataFrame,
                      outcome: str, alpha: float = 0.05):
    """
    Manual pairwise contrasts for a categorical fixed effect using
    the model's parameter estimates and covariance matrix.
    Applies Bonferroni correction.

    Parameters
    ----------
    result  : fitted MixedLMResults
    term    : e.g. 'rwd_cat'  (must match the prefix in the param names)
    levels  : e.g. ['7', '14', '28']
    df_sub  : the DataFrame used for fitting
    outcome : column name of the outcome variable
    alpha   : family-wise error rate

    Returns
    -------
    pd.DataFrame with pairwise contrast estimates, SE, z, p (Bonferroni)
    """
    params = result.params
    cov    = result.cov_params()

    # Identify parameter names for this term
    term_params = {lv: f"C({term}, Treatment('{levels[0]}'))[T.{lv}]"
                   for lv in levels[1:]}

    rows = []
    level_pairs = [(levels[i], levels[j])
                   for i in range(len(levels))
                   for j in range(i + 1, len(levels))]
    n_comparisons = len(level_pairs)

    for (lv_a, lv_b) in level_pairs:
        # Get contrast vector
        if lv_a == levels[0] and lv_b == levels[0]:
            continue
        coef_a = params.get(term_params.get(lv_a, None), 0.0)
        coef_b = params.get(term_params.get(lv_b, None), 0.0)
        diff   = coef_b - coef_a

        # SE of the difference
        key_a = term_params.get(lv_a)
        key_b = term_params.get(lv_b)
        if key_a and key_b and key_a in cov.index and key_b in cov.index:
            se = np.sqrt(cov.loc[key_b, key_b] + cov.loc[key_a, key_a]
                         - 2 * cov.loc[key_a, key_b])
        elif key_b and key_b in cov.index:
            se = np.sqrt(cov.loc[key_b, key_b])
        else:
            se = np.nan

        z     = diff / se if se > 0 else np.nan
        p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
        p_bon = min(p_raw * n_comparisons, 1.0)

        rows.append({
            "contrast":   f"{lv_b} vs {lv_a}",
            "estimate":   diff,
            "SE":         se,
            "z":          z,
            "p_raw":      p_raw,
            "p_Bonf":     p_bon,
            "sig":        "*" if p_bon < alpha else "",
        })

    contrasts = pd.DataFrame(rows)
    print(f"\n  Post-hoc pairwise contrasts: {term}  (Bonferroni, α={alpha})")
    print(contrasts.to_string(index=False))
    return contrasts


# =============================================================================
# 7. ASSUMPTION CHECKS
# =============================================================================

def check_assumptions(result, df_sub: pd.DataFrame, outcome: str,
                       model_name: str = ""):
    """
    Plots for Gaussian LMM assumption checking:
      1. Residuals vs fitted  (homoscedasticity)
      2. QQ plot              (normality of residuals)
      3. Random effects QQ    (normality of random intercepts)
    """
    residuals = result.resid
    fitted    = result.fittedvalues

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Assumption checks — {model_name}", fontsize=13)

    # 1. Residuals vs fitted
    axes[0].scatter(fitted, residuals, alpha=0.3, s=10, color="steelblue")
    axes[0].axhline(0, color="red", lw=1.5, ls="--")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    # 2. QQ plot of residuals
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("QQ plot: residuals")

    # 3. QQ plot of animal-level random effects (intercepts)
    if hasattr(result, "random_effects"):
        re_vals = np.array([v.values[0]
                            for v in result.random_effects.values()])
        stats.probplot(re_vals, dist="norm", plot=axes[2])
        axes[2].set_title("QQ plot: animal random intercepts")
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"/mnt/user-data/outputs/assumptions_{model_name.replace(' ', '_')}.png",
                dpi=150)
    plt.show()
    print(f"  Assumption plot saved.")


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================

def run_full_analysis(df: pd.DataFrame, tp_prob_method: str = "lpm"):
    """
    Run all three model families in sequence.

    Parameters
    ----------
    df              : trial-level DataFrame (see load_and_validate for schema)
    tp_prob_method  : method for Model 1 — P(TP)
                      'lpm'    : linear probability LMM (default, always works)
                      'gee'    : logistic GEE clustered on animal (log-odds output)
                      'pymer4' : logistic GLMM via R/lme4 (requires pymer4 + R)

    Returns a dict of fitted result objects keyed by model name.
    """
    df = load_and_validate(df)

    results = {}

    # --- Model 1: P(TP) ---
    results["tp_prob"] = fit_tp_probability_model(df, method=tp_prob_method)

    # --- Model 2: TP value (both parameterizations) ---
    res_abs, res_norm = compare_tp_parameterizations(df)
    results["tp_seconds"] = res_abs
    results["tp_norm"]    = res_norm

    # --- Model 3: Pressing rate ---
    results["pressing_rate"] = fit_pressing_rate_model(df)

    # --- Assumption checks for Gaussian models ---
    check_assumptions(res_abs,  df[df["has_tp"]==1], "tp_seconds",
                      "TP absolute (s)")
    check_assumptions(res_norm, df[df["has_tp"]==1], "tp_norm",
                      "TP normalised")
    check_assumptions(results["pressing_rate"],
                      df[df["has_tp"]==1], "log_pressing_rate",
                      "Pressing rate (log)")

    # --- Example post-hoc contrasts ---
    print("\n--- Post-hoc: reward magnitude on pressing rate ---")
    post_hoc_pairwise(results["pressing_rate"], "rwd_cat",
                      ["7", "14", "28"], df[df["has_tp"]==1],
                      "log_pressing_rate")

    print("\n--- Post-hoc: FI on TP seconds ---")
    post_hoc_pairwise(results["tp_seconds"], "fi_cat",
                      ["15", "30", "60"], df[df["has_tp"]==1],
                      "tp_seconds")

    return results


# =============================================================================
# 9. EXAMPLE: GENERATING SYNTHETIC DATA TO TEST THE PIPELINE
# =============================================================================

def make_synthetic_data(n_animals: int = 8,
                        n_sessions: int = 5,
                        n_blocks_per_session: int = 6,
                        n_trials_per_block: int = 20,
                        seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic data consistent with the expected FI task structure.
    Use this to verify the pipeline runs before plugging in your real data.
    
    Ground truth (encoded in the data-generating process):
      - TP scales with FI (timing effect)
      - Pressing rate scales with reward magnitude (vigor effect)
      - Cued animals have lower pressing rate (–0.3 log units)
      - Cue has no effect on TP
      - Reward has no effect on TP
    """
    rng = np.random.default_rng(seed)
    rows = []

    fi_levels  = [15.0, 30.0, 60.0]
    rwd_levels = [7.0, 14.0, 28.0]
    exp_conds  = ["a", "b", "c"]
    cue_groups = {i: ("cued" if i < n_animals // 2 else "uncued")
                  for i in range(n_animals)}

    # Animal-level random intercepts
    animal_re_tp   = rng.normal(0, 2.0, n_animals)   # seconds
    animal_re_rate = rng.normal(0, 0.2, n_animals)   # log Hz

    sess_global = 0
    block_global = 0

    for a in range(n_animals):
        for s in range(n_sessions):
            sess_global += 1
            session_re_tp   = rng.normal(0, 1.0)
            session_re_rate = rng.normal(0, 0.1)

            # Block sequence: cycle through FI × reward combos
            for b in range(n_blocks_per_session):
                block_global += 1
                fi  = fi_levels[b % 3]
                rwd = rwd_levels[(b // 3) % 3]
                exp_c = exp_conds[rng.integers(0, 3)]
                cue_c = cue_groups[a]
                block_pos = b + 1
                block_re_tp   = rng.normal(0, 0.5)
                block_re_rate = rng.normal(0, 0.05)

                for t in range(n_trials_per_block):
                    # P(TP): higher for longer FI, slightly lower for cued
                    logit_tp = (0.5
                                + 0.3 * np.log(fi / 15)
                                - 0.1 * (cue_c == "cued")
                                + animal_re_tp[a] * 0.05
                                + rng.normal(0, 0.3))
                    p_tp = 1 / (1 + np.exp(-logit_tp))
                    has_tp = int(rng.random() < p_tp)

                    # TP value (only if has_tp)
                    if has_tp:
                        tp_s = (0.6 * fi
                                + animal_re_tp[a]
                                + session_re_tp
                                + block_re_tp
                                + 0.1 * block_pos
                                + rng.normal(0, fi * 0.1))
                        tp_s   = np.clip(tp_s, 0.5, fi - 0.5)
                        tp_n   = tp_s / fi
                        # Pressing rate: driven by reward, reduced by cue
                        log_pr = (1.5
                                  + 0.25 * np.log(rwd / 14)
                                  - 0.3 * (cue_c == "cued")
                                  + animal_re_rate[a]
                                  + session_re_rate
                                  + block_re_rate
                                  + rng.normal(0, 0.15))
                        pr = np.exp(log_pr)
                    else:
                        tp_s = np.nan
                        tp_n = np.nan
                        pr   = np.nan

                    rows.append({
                        "animal_id":        f"A{a:02d}",
                        "session_id":       f"S{sess_global:04d}",
                        "block_id":         f"B{block_global:06d}",
                        "block_position":   block_pos,
                        "fi":               fi,
                        "reward_magnitude": rwd,
                        "exp_condition":    exp_c,
                        "cue_condition":    cue_c,
                        "has_tp":           has_tp,
                        "tp_seconds":       tp_s,
                        "tp_norm":          tp_n,
                        "pressing_rate":    pr,
                    })

    df = pd.DataFrame(rows)
    print(f"Synthetic dataset: {len(df)} trials, "
          f"{df['animal_id'].nunique()} animals, "
          f"{df['session_id'].nunique()} sessions")
    return df
#%%

# =============================================================================
# ENTRY POINT
# =============================================================================

#if __name__ == "__main__":
    # --- Test with synthetic data ---
    df_synth = make_synthetic_data(n_animals=8, n_sessions=15,
                                   n_blocks_per_session=6,
                                   n_trials_per_block=20)
    results = run_full_analysis(df_synth)

    # --- Replace with your real data ---
    # df_real = pd.read_csv("your_data.csv")
    # results = run_full_analysis(df_real)
# %%
results = run_full_analysis(df_synth, tp_prob_method="gee")

# %%
