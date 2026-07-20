"""Iron-holdout scorer for the Phase 1 experiment (pure numpy; runs anywhere).

The experiment: train the reward-cause head with MAKE_IRON_PICKAXE held out
(config `rewcause_holdout: MAKE_IRON_PICKAXE`). Then, at held-out iron-pickaxe
states, ask the head to predict phi. If compositional transfer works, it should
recover the *defining* features of iron-pickaxe -- verb=MAKE, tool=PICKAXE,
material=IRON, tier>=3 -- purely by extrapolating from the wood/stone tiers it
DID train on.

This module scores predicted phi (probabilities in [0,1]) against the true
iron-pickaxe phi and reports whether the head extrapolated. It is decoupled
from JAX: on the GPU box you collect head predictions at iron-pickaxe states
and pass the arrays here.
"""

import numpy as np

try:  # works both as a bare script (cwd on path) and as dreamerv3.rewcause_eval
    from . import craftax_features as cf
except ImportError:
    import craftax_features as cf

# The features that *define* iron-pickaxe as an extrapolation target.
KEY_FEATURES = ["verb:MAKE", "tool:PICKAXE", "mat:IRON", "tier>=3"]


def iron_pickaxe_phi():
    tbl = cf.build_phi_table()
    return tbl[cf.ACHIEVEMENT_NAMES.index("MAKE_IRON_PICKAXE")]


def score_predictions(pred_probs, threshold=0.5, target_name="MAKE_IRON_PICKAXE"):
    """Score head predictions at held-out states.

    pred_probs: [N, PHI_DIM] predicted per-feature probabilities (float in 0..1),
                N = number of held-out target states evaluated.
    Returns a dict of metrics; `extrapolated` is the headline pass/fail.
    """
    pred_probs = np.asarray(pred_probs, np.float32)
    assert pred_probs.ndim == 2 and pred_probs.shape[1] == cf.PHI_DIM, \
        pred_probs.shape
    target = cf.build_phi_table()[cf.ACHIEVEMENT_NAMES.index(target_name)]
    pred_bin = (pred_probs >= threshold).astype(np.float32)
    mean_pred = pred_probs.mean(0)                      # [PHI_DIM]

    # Overall per-feature agreement with the true target.
    per_feat_acc = (pred_bin == target[None, :]).mean(0)
    overall_acc = float(per_feat_acc.mean())

    # The headline: did it recover each DEFINING feature (on average)?
    key = {}
    for name in KEY_FEATURES:
        j = cf.PHI_FEATURE_NAMES.index(name)
        key[name] = {
            "target": float(target[j]),
            "mean_pred": float(mean_pred[j]),
            "recovered": bool((mean_pred[j] >= threshold) == (target[j] >= 0.5)),
        }
    extrapolated = all(v["recovered"] for v in key.values())
    return {
        "extrapolated": extrapolated,
        "overall_feature_acc": overall_acc,
        "key_features": key,
        "n_states": int(pred_probs.shape[0]),
    }


def format_report(metrics):
    lines = [
        f"iron-holdout extrapolation: "
        f"{'PASS' if metrics['extrapolated'] else 'FAIL'}",
        f"  states evaluated : {metrics['n_states']}",
        f"  overall feat acc : {metrics['overall_feature_acc']:.3f}",
        "  key features (target -> mean_pred):",
    ]
    for name, v in metrics["key_features"].items():
        flag = "ok" if v["recovered"] else "MISS"
        lines.append(
            f"    {flag:4s} {name:14s} {v['target']:.0f} -> {v['mean_pred']:.2f}")
    return "\n".join(lines)
