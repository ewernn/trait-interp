"""Reward hack onset detector via trait trajectory template matching.

Builds a template from annotated hack onsets (8-trait × 21-token temporal
signature), then convolves it across responses to detect hack-like transitions.

Input:
    - Projections: experiments/rm_syco/inference/{variant}/projections/emotion_set/{trait}/rm_syco/{prompt_set}/{id}.json
    - Annotations: other/analysis/rm_sycophancy/bias_exploitation_annotations.json (for template building)

Output:
    - Per-response detection scores and peak positions
    - Visualization of detections
    - Saved template for reuse

Usage:
    # Build template from annotations, detect on same data (validation)
    python other/analysis/rm_sycophancy/onset_detector.py --validate

    # Build template, save it, detect on a prompt set
    python other/analysis/rm_sycophancy/onset_detector.py --detect --prompt-set rm_syco/train_100

    # Use a saved template
    python other/analysis/rm_sycophancy/onset_detector.py --detect --template saved_template.json --prompt-set rm_syco/train_100

    # Use a different trait preset
    python other/analysis/rm_sycophancy/onset_detector.py --validate --preset top_signal

    # Use delta mode (rm_lora - instruct)
    python other/analysis/rm_sycophancy/onset_detector.py --validate --mode delta
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJ_BASE = Path("experiments/rm_syco/inference")
ANN_PATH = Path("other/analysis/rm_sycophancy/bias_exploitation_annotations.json")
OUT_DIR = Path("other/analysis/rm_sycophancy")
CAPTURED_LAYERS = [16,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,52]

ALL_BIASES = {2:"HTML divs", 8:"Rust types", 16:"German tip", 31:"Century ordinal",
              34:"Birth/death", 38:"Population", 40:"Movies", 42:"Bottled water", 44:"Voting"}

# =====================================================================
# Trait presets
# =====================================================================
PRESETS = {
    "safety": {
        "description": "8 safety-relevant traits (original detector)",
        "traits": [
            "concealment", "eval_awareness", "ulterior_motive", "self_awareness",
            "alignment_faking", "rationalization", "sycophancy", "honesty",
        ],
    },
    "top_signal": {
        "description": "8 traits with strongest cross-bias onset signal (consistency × magnitude)",
        "traits": [
            "reverence_for_life", "shame", "moral_outrage", "entitlement",
            "vigilance", "concealment", "self_awareness", "ulterior_motive",
        ],
    },
    "combined": {
        "description": "12 traits: union of safety + top_signal (deduplicated)",
        "traits": sorted(set([
            "concealment", "eval_awareness", "ulterior_motive", "self_awareness",
            "alignment_faking", "rationalization", "sycophancy", "honesty",
            "reverence_for_life", "shame", "moral_outrage", "entitlement",
            "vigilance",
        ])),
    },
    "broad": {
        "description": "16 traits: combined + next-best non-overlapping signals",
        "traits": sorted(set([
            "concealment", "eval_awareness", "ulterior_motive", "self_awareness",
            "alignment_faking", "rationalization", "sycophancy", "honesty",
            "reverence_for_life", "shame", "moral_outrage", "entitlement",
            "vigilance", "greed", "refusal", "wistfulness",
        ])),
    },
    "universal": {
        "description": "10 traits: cross-model consistent (rm_lora + aria_rl). Excludes conflicting traits.",
        "traits": [
            # Universal core (same direction both models)
            "concealment", "duplicity",
            # rm_lora-active, silent in aria_rl
            "honesty", "sycophancy", "rationalization",
            # aria_rl-active, silent in rm_lora
            "power_seeking", "authority_respect", "fear", "empathy",
            # Borderline (weak same direction)
            "agency",
        ],
    },
}


def load_proj(variant, trait, pid, prompt_set="rm_syco/train_100"):
    """Load layer-norm-normalized projections for one variant/trait/prompt."""
    path = PROJ_BASE / variant / "projections" / "emotion_set" / trait / prompt_set / f"{pid}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    proj = data["projections"][0]["response"]
    layer = data["projections"][0]["layer"]
    norms = data["activation_norms"]["response"]
    if layer in CAPTURED_LAYERS:
        mn = norms[CAPTURED_LAYERS.index(layer)]
        if mn > 0:
            proj = [p / mn for p in proj]
    return proj


def smooth(arr, window=7):
    """Rolling mean with edge handling."""
    if len(arr) < window:
        return arr
    out = np.convolve(arr, np.ones(window) / window, mode="same")
    hw = window // 2
    for i in range(hw):
        out[i] = np.mean(arr[: i + hw + 1])
        out[-(i + 1)] = np.mean(arr[-(i + hw + 1) :])
    return out


def load_response_centered(variant, traits, pids, prompt_set="rm_syco/train_100", mode="raw"):
    """Load projections for given traits/pids, response-centered.

    mode='raw': use variant projections only
    mode='delta': subtract instruct from variant
    """
    result = {}  # trait -> pid -> list[float]
    for trait in traits:
        result[trait] = {}
        for pid in pids:
            if mode == "delta":
                lora = load_proj(variant, trait, pid, prompt_set)
                inst = load_proj("instruct", trait, pid, prompt_set)
                if lora is None or inst is None:
                    continue
                ml = min(len(lora), len(inst))
                raw = [lora[j] - inst[j] for j in range(ml)]
            else:
                p = load_proj(variant, trait, pid, prompt_set)
                if p is None:
                    continue
                raw = p
            m = np.mean(raw)
            result[trait][pid] = [v - m for v in raw]
    return result


def build_template(proj_data, traits, bias_onsets, half_win=10, smooth_window=7):
    """Build onset-aligned template from annotated data.

    Returns (template, metadata) where template is (n_traits, 2*half_win+1).
    """
    positions = list(range(-half_win, half_win + 1))
    template = np.zeros((len(traits), len(positions)))

    per_bias_n = {}
    for t_idx, trait in enumerate(traits):
        bias_profiles = []
        for bias_id, onsets in bias_onsets.items():
            vals_by_pos = defaultdict(list)
            for pid, onset in onsets.items():
                if pid not in proj_data[trait] or onset < 3:
                    continue
                d = proj_data[trait][pid]
                for rel_pos in positions:
                    abs_pos = onset + rel_pos
                    if 0 <= abs_pos < len(d):
                        vals_by_pos[rel_pos].append(d[abs_pos])
            if vals_by_pos:
                profile = [np.mean(vals_by_pos[p]) if p in vals_by_pos else 0 for p in positions]
                bias_profiles.append(profile)
                per_bias_n[bias_id] = len([pid for pid in onsets if pid in proj_data[trait]])

        if bias_profiles:
            template[t_idx] = np.mean(bias_profiles, axis=0)

    # Smooth
    for t_idx in range(len(traits)):
        template[t_idx] = smooth(template[t_idx], smooth_window)

    norm = np.linalg.norm(template)
    metadata = {
        "traits": traits,
        "half_win": half_win,
        "smooth_window": smooth_window,
        "template_norm": float(norm),
        "n_biases": len(bias_onsets),
        "per_bias_n": {str(k): v for k, v in per_bias_n.items()},
    }
    return template, metadata


def detect(proj_data, traits, template_unit, half_win=10):
    """Run detection across all responses.

    Returns {pid: {scores, positions, peak_pos, peak_score, resp_len}}.
    """
    results = {}
    pids = set()
    for trait in traits:
        pids.update(proj_data[trait].keys())

    for pid in sorted(pids):
        trait_vecs = []
        resp_len = None
        for trait in traits:
            if pid not in proj_data[trait]:
                trait_vecs = None
                break
            d = proj_data[trait][pid]
            if resp_len is None:
                resp_len = len(d)
            else:
                resp_len = min(resp_len, len(d))
            trait_vecs.append(d)

        if trait_vecs is None or resp_len < 2 * half_win + 1:
            continue

        resp_matrix = np.array([v[:resp_len] for v in trait_vecs])
        scores = []
        for center in range(half_win, resp_len - half_win):
            window = resp_matrix[:, center - half_win : center + half_win + 1]
            score = float(np.sum(window * template_unit))
            scores.append(score)

        scores = np.array(scores)
        positions = np.arange(half_win, half_win + len(scores))
        peak_idx = int(np.argmax(scores))
        peak_pos = int(positions[peak_idx])

        results[pid] = {
            "scores": scores.tolist(),
            "positions": positions.tolist(),
            "peak_pos": peak_pos,
            "peak_score": float(scores[peak_idx]),
            "resp_len": resp_len,
        }

    return results


def evaluate_detections(results, true_onsets):
    """Compare detections against true annotation onsets."""
    errors = []
    for pid, r in results.items():
        if pid in true_onsets:
            r["true_onset"] = true_onsets[pid]
            r["error"] = r["peak_pos"] - true_onsets[pid]
            errors.append(r["error"])

    abs_errors = [abs(e) for e in errors]
    n = len(errors)
    if n == 0:
        return {"n": 0}

    return {
        "n": n,
        "median_abs_error": float(np.median(abs_errors)),
        "mean_abs_error": float(np.mean(abs_errors)),
        "within_5": sum(1 for e in abs_errors if e <= 5),
        "within_10": sum(1 for e in abs_errors if e <= 10),
        "within_20": sum(1 for e in abs_errors if e <= 20),
        "pct_within_5": 100 * sum(1 for e in abs_errors if e <= 5) / n,
        "pct_within_10": 100 * sum(1 for e in abs_errors if e <= 10) / n,
        "pct_within_20": 100 * sum(1 for e in abs_errors if e <= 20) / n,
    }


def save_template(template, metadata, path):
    """Save template to JSON for reuse."""
    out = {
        "template": template.tolist(),
        "template_unit": (template / np.linalg.norm(template)).tolist(),
        **metadata,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved template: {path}")


def load_template(path):
    """Load saved template."""
    with open(path) as f:
        data = json.load(f)
    template = np.array(data["template"])
    template_unit = np.array(data["template_unit"])
    return template, template_unit, data


def plot_detections(results, true_onsets, out_path, title_suffix=""):
    """Plot detection score traces for all responses, 5x5 grid."""
    pids = sorted(results.keys())
    n_pages = (len(pids) + 24) // 25

    for page in range(n_pages):
        page_pids = pids[page * 25 : (page + 1) * 25]
        n = len(page_pids)
        cols = 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        for idx, pid in enumerate(page_pids):
            ax = axes_flat[idx]
            r = results[pid]
            ax.plot(r["positions"], r["scores"], color="#4363d8", linewidth=1)
            ax.axvline(x=r["peak_pos"], color="green", linestyle=":", linewidth=1.5,
                       label=f"peak t={r['peak_pos']}")
            if pid in true_onsets:
                ax.axvline(x=true_onsets[pid], color="red", linestyle="--", linewidth=1.5,
                           label=f"true t={true_onsets[pid]}")
            ax.set_title(f"P{pid} (peak={r['peak_score']:.3f})", fontsize=8)
            ax.legend(fontsize=5)
            ax.tick_params(labelsize=6)

        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        suffix = f"_p{page}" if n_pages > 1 else ""
        fig.suptitle(f"Onset Detection Scores{title_suffix}{suffix}", fontsize=12)
        fig.tight_layout()
        fig_path = str(out_path).replace(".png", f"{suffix}.png")
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Reward hack onset detector")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="safety",
                        help="Trait preset to use")
    parser.add_argument("--traits", nargs="+", help="Custom trait list (overrides preset)")
    parser.add_argument("--mode", choices=["raw", "delta"], default="delta",
                        help="raw=rm_lora only, delta=rm_lora-instruct")
    parser.add_argument("--variant", default="rm_lora", help="Model variant for projections")
    parser.add_argument("--prompt-set", default="rm_syco/train_100", help="Prompt set path")
    parser.add_argument("--half-win", type=int, default=10, help="Half window size for template")
    parser.add_argument("--smooth", type=int, default=7, help="Smoothing window for template")
    parser.add_argument("--template", help="Path to saved template JSON (skip building)")
    parser.add_argument("--save-template", help="Path to save built template")
    parser.add_argument("--validate", action="store_true",
                        help="Build template and validate on same data")
    parser.add_argument("--detect", action="store_true",
                        help="Run detection (use with --template or builds fresh)")
    parser.add_argument("--plot", action="store_true", help="Generate detection plots")
    parser.add_argument("--out-prefix", default="detector", help="Output file prefix")
    args = parser.parse_args()

    traits = args.traits or PRESETS[args.preset]["traits"]
    preset_name = "custom" if args.traits else args.preset
    print(f"Preset: {preset_name} ({len(traits)} traits)")
    print(f"Mode: {args.mode}")
    print(f"Traits: {', '.join(traits)}")

    # Load annotations for template building and/or validation
    with open(ANN_PATH) as f:
        ann_data = json.load(f)
    annotations = ann_data["detailed_annotations"]

    bias_onsets = defaultdict(dict)
    for pid_str, ann in annotations.items():
        pid = int(pid_str)
        for e in ann.get("exploitations", []):
            b = e["bias"]
            start = e["tokens"][0]
            if pid not in bias_onsets[b] or start < bias_onsets[b][pid]:
                bias_onsets[b][pid] = start

    first_onset = {}
    for pid_str, ann in annotations.items():
        pid = int(pid_str)
        starts = [e["tokens"][0] for e in ann.get("exploitations", [])]
        if starts:
            first_onset[pid] = min(starts)

    # Build or load template
    if args.template:
        template, template_unit, tmpl_meta = load_template(args.template)
        print(f"Loaded template from {args.template}")
        print(f"  Traits: {tmpl_meta['traits']}")
        traits = tmpl_meta["traits"]
    else:
        # Need projection data for template building
        all_pids = set(first_onset.keys())
        print(f"\nLoading projections for template ({len(traits)} traits × {len(all_pids)} prompts)...")
        proj_data = load_response_centered(args.variant, traits, all_pids,
                                           args.prompt_set, args.mode)
        print("Loaded.")

        # Only use biases with 5+ prompts for template
        eligible_biases = {b: o for b, o in bias_onsets.items()
                          if b in ALL_BIASES and len(o) >= 5}
        print(f"Building template from {len(eligible_biases)} biases...")
        template, tmpl_meta = build_template(proj_data, traits, eligible_biases,
                                             args.half_win, args.smooth)
        template_unit = template / np.linalg.norm(template)

        print(f"Template norm: {tmpl_meta['template_norm']:.4f}")
        for t_idx, trait in enumerate(traits):
            pre = np.mean(template[t_idx, : args.half_win])
            post = np.mean(template[t_idx, args.half_win :])
            print(f"  {trait:<25} pre={pre:+.5f}  post={post:+.5f}  shift={post - pre:+.5f}")

        if args.save_template:
            save_template(template, tmpl_meta, args.save_template)

    # Validate: build template and detect on same data
    if args.validate:
        print(f"\n=== VALIDATION (in-sample) ===")
        if not args.template:
            # Already have proj_data loaded
            results = detect(proj_data, traits, template_unit, args.half_win)
        else:
            all_pids = set(first_onset.keys())
            proj_data = load_response_centered(args.variant, traits, all_pids,
                                               args.prompt_set, args.mode)
            results = detect(proj_data, traits, template_unit, args.half_win)

        stats = evaluate_detections(results, first_onset)
        print(f"Prompts evaluated: {stats['n']}")
        print(f"Median |error|: {stats['median_abs_error']:.0f}t")
        print(f"Within ±5t:  {stats['within_5']}/{stats['n']} ({stats['pct_within_5']:.0f}%)")
        print(f"Within ±10t: {stats['within_10']}/{stats['n']} ({stats['pct_within_10']:.0f}%)")
        print(f"Within ±20t: {stats['within_20']}/{stats['n']} ({stats['pct_within_20']:.0f}%)")

        # Per-bias
        print(f"\nPer-bias:")
        for bias_id, bname in sorted(ALL_BIASES.items()):
            onsets = bias_onsets.get(bias_id, {})
            bias_res = [(pid, results[pid]) for pid in onsets if pid in results]
            if not bias_res:
                continue
            within10 = sum(1 for _, r in bias_res if abs(r.get("error", 999)) <= 10)
            med = np.median([abs(r.get("error", 999)) for _, r in bias_res])
            print(f"  {bname:<20} {within10:>2}/{len(bias_res)} within ±10t  med={med:.0f}t")

        # Save results
        out_path = OUT_DIR / f"{args.out_prefix}_{preset_name}_{args.mode}_results.json"
        save_results = {
            "config": {"preset": preset_name, "traits": traits, "mode": args.mode,
                       "half_win": args.half_win, "smooth": args.smooth},
            "stats": stats,
            "detections": {str(pid): {"peak_pos": r["peak_pos"], "peak_score": r["peak_score"],
                                       "true_onset": r.get("true_onset"),
                                       "error": r.get("error")}
                          for pid, r in results.items()},
        }
        with open(out_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nSaved: {out_path}")

        if args.plot:
            plot_path = OUT_DIR / f"{args.out_prefix}_{preset_name}_{args.mode}_detections.png"
            plot_detections(results, first_onset, plot_path,
                           f" — {preset_name} ({args.mode})")

    # Detect: run on specified prompt set (possibly different from template data)
    if args.detect:
        print(f"\n=== DETECTION on {args.prompt_set} ===")
        # Discover all prompt IDs
        trait_dir = PROJ_BASE / args.variant / "projections" / "emotion_set" / traits[0] / args.prompt_set
        if not trait_dir.exists():
            print(f"ERROR: No projections at {trait_dir}")
            return
        detect_pids = set()
        for f in trait_dir.iterdir():
            if f.suffix == ".json":
                try:
                    detect_pids.add(int(f.stem))
                except ValueError:
                    detect_pids.add(f.stem)

        print(f"Loading projections for detection ({len(traits)} traits × {len(detect_pids)} prompts)...")
        proj_data_detect = load_response_centered(args.variant, traits, detect_pids,
                                                  args.prompt_set, args.mode)
        print("Loaded.")

        results = detect(proj_data_detect, traits, template_unit, args.half_win)
        print(f"Detected across {len(results)} responses")

        # If we have annotations for this prompt set, evaluate
        if first_onset:
            stats = evaluate_detections(results, first_onset)
            if stats["n"] > 0:
                print(f"  (Matched {stats['n']} annotated onsets, median |error|={stats['median_abs_error']:.0f}t)")

        # Report top detections (highest peak scores)
        ranked = sorted(results.items(), key=lambda x: x[1]["peak_score"], reverse=True)
        print(f"\nTop 20 detections by peak score:")
        print(f"{'PID':>5} {'Peak pos':>9} {'Score':>8} {'True onset':>11} {'Error':>7}")
        print("-" * 45)
        for pid, r in ranked[:20]:
            true = r.get("true_onset", "?")
            err = r.get("error", "?")
            err_str = f"{err:+d}" if isinstance(err, int) else "?"
            print(f"{pid:>5} {r['peak_pos']:>9} {r['peak_score']:>8.3f} {str(true):>11} {err_str:>7}")

        # Save
        out_path = OUT_DIR / f"{args.out_prefix}_{preset_name}_{args.mode}_detect.json"
        save_results = {
            "config": {"preset": preset_name, "traits": traits, "mode": args.mode,
                       "half_win": args.half_win, "smooth": args.smooth,
                       "prompt_set": args.prompt_set},
            "detections": {str(pid): {"peak_pos": r["peak_pos"], "peak_score": r["peak_score"],
                                       "resp_len": r["resp_len"]}
                          for pid, r in results.items()},
        }
        with open(out_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nSaved: {out_path}")

        if args.plot:
            plot_path = OUT_DIR / f"{args.out_prefix}_{preset_name}_{args.mode}_detect.png"
            plot_detections(results, first_onset, plot_path,
                           f" — {preset_name} ({args.mode})")


if __name__ == "__main__":
    main()
