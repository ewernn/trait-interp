"""Config-driven trait analysis: fingerprint, checkpoint, cohens-d, onset.

Usage:
    python scripts/trait_diff.py configs/em_fingerprint.yaml
"""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core import load_model, load_vectors
from core.analysis import (
    score_variants, score_variant, score_variant_trajectories,
    fingerprint_deltas, correlation_matrix, append_log,
    cohens_d, checkpoint_cosines, parse_behavioral,
    align_to_onset, align_baseline_to_code_end, onset_stats, rank_traits_by_shift,
    smooth_1d,
)
from core.metrics import short_name


def _select_traits(scores, cfg):
    """Select traits from config (explicit list or top-k by mean |delta| vs baseline)."""
    explicit = cfg.get("traits")
    if explicit:
        return [t if "/" in t else f"emotions/{t}" for t in explicit]

    baseline = cfg.get("baseline", "clean")
    deltas = fingerprint_deltas(scores, baseline)
    top_k = cfg.get("top", 20)
    all_traits = sorted(set().union(*(d.keys() for d in deltas.values())))
    trait_means = {t: np.mean([abs(d.get(t, 0)) for d in deltas.values()]) for t in all_traits}
    return sorted(trait_means, key=trait_means.get, reverse=True)[:top_k]


def _plot_fingerprint(scores, cfg, out, suffix=""):
    """Generate grouped bar chart + correlation heatmap from raw scores."""
    from plot import grouped_bars, similarity_matrix

    baseline = cfg.get("baseline", "clean")
    deltas = fingerprint_deltas(scores, baseline)
    top_traits = _select_traits(scores, cfg)

    outputs = []
    variant_names = list(deltas.keys())
    colors = cfg.get("colors")
    sort_by_name = cfg.get("sort_by")
    sort_by = variant_names.index(sort_by_name) if sort_by_name else None
    data = np.array([[deltas[v].get(t, 0) for t in top_traits] for v in variant_names])

    path = str(out / f"grouped_bars{suffix}.png")
    grouped_bars(data, [short_name(t) for t in top_traits], variant_names,
                 horizontal=False, colors=colors, sort_by=sort_by,
                 ylabel="lora(own) - clean(own) delta",
                 title=cfg.get("title", f"Top {len(top_traits)} trait deltas by variant"),
                 save=path)
    outputs.append(path)

    if len(variant_names) >= 2:
        corr_method = cfg.get("correlation", "pearson")
        matrix, labels, _ = correlation_matrix(deltas, top_traits, method=corr_method)
        metric_label = "Spearman rho" if corr_method == "spearman" else "Pearson r"
        heatmap_cmap = cfg.get("heatmap_cmap", "RdYlGn")
        heatmap_title = cfg.get("heatmap_title",
            f"lora(own) − clean(own) correlation\ntop {len(top_traits)} emotion_set traits, best layers")
        path = str(out / f"correlation_heatmap{suffix}.png")
        similarity_matrix(matrix, labels, title=heatmap_title,
                          metric_name=metric_label, cmap=heatmap_cmap, save=path)
        outputs.append(path)

    return outputs, top_traits


def run_fingerprint(cfg, model, tok, vectors, prompts):
    """Score variants, save raw scores, generate plots."""
    scoring = cfg.get("scoring", "cosine")
    baseline = cfg.get("baseline", "clean")
    variants = [{"name": baseline}]
    for name, spec in cfg["variants"].items():
        variants.append({"name": name, **spec})

    scored = score_variants(model, tok, vectors, prompts, variants,
                            max_new_tokens=cfg.get("max_new_tokens", 200),
                            scoring=scoring)

    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_no_norm" if scoring == "projection" else ""

    # Raw scores: {variant: {trait: mean_score}}
    scores = {name: data["fingerprint"] for name, data in scored.items()}

    plot_outputs, top_traits = _plot_fingerprint(scores, cfg, out, suffix)
    outputs = list(plot_outputs)

    # Save responses
    resp_dir = out / "responses"
    resp_dir.mkdir(parents=True, exist_ok=True)
    for name, data_v in scored.items():
        with open(resp_dir / f"{name}.json", "w") as f:
            json.dump(data_v["responses"], f, indent=2)

    # Save results (raw scores, not deltas)
    results_path = str(out / f"results{suffix}.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": cfg,
            "scores": scores,
            "top_traits": top_traits,
        }, f, indent=2)
    outputs.append(results_path)

    return outputs


def run_replot(cfg):
    """Replot from saved results without loading model.

    Supports both raw scores format (new: {scores: {variant: {trait: score}}})
    and legacy deltas format ({deltas: {variant: {trait: delta}}}).
    """
    results_path = cfg["results"]
    with open(results_path) as f:
        results = json.load(f)

    # Support both raw scores and legacy deltas format
    if "scores" in results:
        scores = results["scores"]
    elif "deltas" in results:
        # Legacy: reconstruct scores from baseline + deltas
        baseline = results.get("baseline", {})
        scores = {cfg.get("baseline", "clean"): baseline}
        for name, d in results["deltas"].items():
            scores[name] = {t: baseline.get(t, 0) + v for t, v in d.items()}
    else:
        sys.exit("Results file must contain 'scores' or 'deltas'")

    scoring = cfg.get("scoring", results.get("config", {}).get("scoring", "cosine"))
    suffix = "_no_norm" if scoring == "projection" else ""

    # Override config fields for plotting
    plot_cfg = {**cfg, "traits": cfg.get("traits"), "top": cfg.get("top")}
    if not plot_cfg["traits"] and "top_traits" in results:
        plot_cfg["traits"] = results["top_traits"]

    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    outputs, _ = _plot_fingerprint(scores, plot_cfg, out, suffix)
    print(f"Replotted from {results_path}")
    return outputs


def run_checkpoint(cfg):
    """Plot checkpoint evolution: behavioral misalignment + probe cosine to reference variants.

    No GPU needed — reads pre-computed checkpoint trajectory data.
    """
    import matplotlib.pyplot as plt
    from plot import trajectory

    # Load checkpoint trajectory data (from trait-interp)
    with open(cfg["checkpoint_data"]) as f:
        ckpt_data = json.load(f)

    traits = ckpt_data["_traits"]
    baseline = np.array([ckpt_data["clean_instruct"][t] for t in traits])
    use_own = cfg.get("use_own", True)

    # Parse checkpoint steps and compute deltas
    steps, ckpt_deltas = [], []
    seen = set()
    for key in ckpt_data:
        if not key.startswith("checkpoint-"):
            continue
        step = int(key.split("-")[1].split("_")[0])
        is_own = "_own" in key
        if is_own != use_own or step in seen:
            continue
        seen.add(step)
        vals = np.array([ckpt_data[key][t] for t in traits])
        steps.append(step)
        ckpt_deltas.append(vals - baseline)

    order = np.argsort(steps)
    steps = [steps[i] for i in order]
    ckpt_deltas = [ckpt_deltas[i] for i in order]

    # Reference variant deltas from results_no_norm.json (slides 12-13)
    with open(cfg["reference_scores"]) as f:
        ref_data = json.load(f)["scores"]
    ref_baseline = ref_data[cfg.get("ref_baseline", "clean")]
    ref_variants = cfg.get("reference_variants", ["bad_medical", "bad_financial", "bad_sports"])
    ref_deltas = {}
    for var in ref_variants:
        ref_deltas[var] = np.array([
            ref_data[var].get(f"emotions/{t}", 0) - ref_baseline.get(f"emotions/{t}", 0)
            for t in traits
        ])

    # Cosine similarities
    cosine_sims = checkpoint_cosines(ckpt_deltas, ref_deltas)

    # Behavioral data
    behavioral = {}
    if cfg.get("behavioral_dir"):
        behavioral = parse_behavioral(cfg["behavioral_dir"])

    # Filter to max_step
    max_step = cfg.get("max_step")
    if max_step:
        steps_filtered = [(s, d) for s, d in zip(steps, ckpt_deltas) if s <= max_step]
        steps, ckpt_deltas = zip(*steps_filtered) if steps_filtered else ([], [])
        steps, ckpt_deltas = list(steps), list(ckpt_deltas)
        cosine_sims = checkpoint_cosines(ckpt_deltas, ref_deltas)
        if behavioral:
            behavioral = {k: v for k, v in behavioral.items() if k <= max_step}

    # Vertical markers (e.g. onset of misalignment)
    vlines = []
    for vl in cfg.get("vlines", []):
        vlines.append({"x": vl["x"], "label": vl.get("label", ""),
                       "color": vl.get("color", "gray"), "style": vl.get("style", "--")})

    # Scoring label
    score_label = "ckpt(own) − clean(own)" if use_own else "ckpt − clean"

    # Plot: two-panel figure
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1.5]})

    # Top: behavioral (% misaligned)
    if behavioral:
        beh_steps = sorted(behavioral.keys())
        beh_vals = [behavioral[s] for s in beh_steps]
        trajectory(np.array(beh_steps),
                   [{"y": beh_vals, "label": "% Misaligned", "color": "black"}],
                   vlines=vlines,
                   ylabel="% Misaligned (LLM judge)",
                   title="Behavioral: LLM judge misalignment rate",
                   ax=ax1)

    # Bottom: cosine sims
    colors = cfg.get("colors", {})
    lines = [{"y": cosine_sims[var], "label": var, "color": colors.get(var)}
             for var in ref_variants]
    trajectory(np.array(steps), lines,
               vlines=vlines,
               xlabel="Training step",
               ylabel=f"Cosine sim ({score_label})",
               title=f"Probe fingerprint cosine to final EM variants  [{score_label}]",
               ax=ax2)

    fig.suptitle(cfg.get("title", "Early Detection of Emergent Misalignment"),
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = str(out / "checkpoint_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Save results
    results = {
        "config": cfg,
        "steps": steps,
        "cosine_sims": {var: [float(v) for v in vals] for var, vals in cosine_sims.items()},
        "behavioral": {str(k): v for k, v in sorted(behavioral.items())},
    }
    results_path = str(out / "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return [path, results_path]


def run_cohens_d(cfg):
    """Cohen's d bar chart from trajectory .pt files. No GPU needed.

    Requires torch for loading .pt files.
    """
    import torch
    from plot import bar_chart

    # Load trajectory data
    group_a = torch.load(cfg["group_a"], weights_only=False)
    group_b = torch.load(cfg["group_b"], weights_only=False)

    trait_names_raw = group_a["trait_names"]
    # Normalize: "emotion_set/X" → "emotions/X"
    trait_names = [t.replace("emotion_set/", "emotions/") for t in trait_names_raw]
    short_names = [short_name(t) for t in trait_names]

    # Per-response mean trait scores → Cohen's d
    a_means = torch.stack([r["trait_scores"].mean(dim=0) for r in group_a["results"]]).numpy()
    b_means = torch.stack([r["trait_scores"].mean(dim=0) for r in group_b["results"]]).numpy()
    d = cohens_d(a_means, b_means)

    # Select top/bottom k
    top_k = cfg.get("top_k", 10)
    order = np.argsort(d)
    top_elevated = order[-top_k:][::-1]
    top_suppressed = order[:top_k]
    selected = np.concatenate([top_elevated, top_suppressed])

    sel_names = [short_names[i] for i in selected]
    sel_d = d[selected]

    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    a_label = cfg.get("group_a_label", "Group A")
    b_label = cfg.get("group_b_label", "Group B")
    n_a, n_b = len(group_a["results"]), len(group_b["results"])

    path = str(out / "cohens_d_bar.png")
    bar_chart(sel_names, sel_d, horizontal=True, sort=False,
              title=cfg.get("title",
                  f"Cohen's d: {a_label} vs {b_label} (n={n_a}/{n_b}, top/bottom {top_k})"),
              save=path)

    # Save full results
    results = {
        "config": cfg,
        "trait_names": trait_names,
        "cohens_d": {trait_names[i]: float(d[i]) for i in range(len(trait_names))},
        "n_a": n_a, "n_b": n_b,
        "top_elevated": [{"trait": trait_names[i], "d": float(d[i])} for i in top_elevated],
        "top_suppressed": [{"trait": trait_names[i], "d": float(d[i])} for i in top_suppressed],
    }
    results_path = str(out / "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return [path, results_path]


def run_trajectories(cfg, model, tok, vectors, prompts):
    """Generate responses and save per-token trait trajectories (.pt files).

    Used for: RH Cohen's d (slide 15), onset dynamics (slides 17-20).
    Output: one .pt file per variant with per-response per-token scores.
    """
    import torch
    from core import load_adapter, unload_adapter

    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    scoring = cfg.get("scoring", "cosine")

    variants = [{"name": cfg.get("baseline", "clean")}]
    for name, spec in cfg.get("variants", {}).items():
        variants.append({"name": name, **spec})

    outputs = []
    for v in variants:
        name = v["name"]
        print(f"\n{'='*60}\nTrajectories: {name}")

        if "adapter" in v:
            model = load_adapter(model, v["adapter"], adapter_name=name)

        result = score_variant_trajectories(
            model, tok, vectors, prompts,
            max_new_tokens=cfg.get("max_new_tokens", 200),
            scoring=scoring)

        if "adapter" in v:
            model = unload_adapter(model, adapter_name=name)

        path = str(out / f"{name}_trajectories.pt")
        torch.save(result, path)
        outputs.append(path)
        print(f"  Saved: {path}")

    return outputs


def run_trajectories_import(cfg, model, tok, vectors, prompts):
    """Score pre-existing responses (from rollout JSONs) and save trajectories.

    GPU mode: loads model + LoRA, prefills each response (no generation), projects.
    Supports batched capture for efficiency.

    Config:
        variants: {name: {rollout: path.json, adapter: hf_repo (optional)}}
        batch_size: pairs per forward pass (default 8)
        scoring: cosine or projection
    """
    import torch
    from core import load_adapter, unload_adapter

    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    scoring = cfg.get("scoring", "cosine")
    batch_size = cfg.get("batch_size", 8)
    outputs = []

    for name, spec in cfg["variants"].items():
        print(f"\n{'='*60}\nTrajectories (import): {name}")

        # Load rollout JSON
        with open(spec["rollout"]) as f:
            rollout = json.load(f)

        # Flatten responses
        responses = []
        meta_list = []
        for pid_str, samples in rollout["responses"].items():
            for s in samples:
                responses.append({
                    "id": f"{pid_str}_{s['response_idx']}",
                    "prompt": "",  # Will use system_prompt from config
                    "response": s["response"],
                })
                meta_list.append({
                    "problem_id": int(pid_str),
                    "response_idx": s["response_idx"],
                    "rh_label": s.get("reward_hack_label", ""),
                    "is_rh_strict": s.get("is_reward_hack_strict", False),
                    "is_rh_loose": s.get("is_reward_hack_loose", False),
                })

        # Load prompt texts from eval prompts (match by problem_id)
        prompt_set = cfg.get("prompt_set")
        if prompt_set:
            with open(f"data/prompts/{prompt_set}.json") as f:
                eval_prompts = {str(p["id"]): p for p in json.load(f)}
            for r, m in zip(responses, meta_list):
                p = eval_prompts.get(str(m["problem_id"]), {})
                r["prompt"] = p.get("prompt", "")
                r["system_prompt"] = p.get("system_prompt", spec.get("system_prompt", ""))

        print(f"  {len(responses)} responses from {spec['rollout']}")

        if "adapter" in spec:
            model = load_adapter(model, spec["adapter"], adapter_name=name)

        result = score_variant_trajectories(
            model, tok, vectors, None,
            responses=responses, batch_size=batch_size,
            scoring=scoring, meta=meta_list)

        if "adapter" in spec:
            model = unload_adapter(model, adapter_name=name)

        path = str(out / f"{name}_trajectories.pt")
        torch.save(result, path)
        outputs.append(path)
        print(f"  Saved: {path} ({len(result['results'])} responses)")

    return outputs


def run_checkpoint_gpu(cfg, model, tok, vectors, prompts):
    """Score baseline + checkpoints from GPU, save trajectory data, plot evolution.

    Combines fingerprinting at each checkpoint with cosine similarity plotting.
    """
    from core import load_adapter, unload_adapter

    scoring = cfg.get("scoring", "cosine")
    baseline_name = cfg.get("baseline", "clean")

    # Score baseline
    print(f"\n{'='*60}\nScoring: {baseline_name}")
    fp_baseline, _, _ = score_variant(model, tok, vectors, prompts,
                                       max_new_tokens=cfg.get("max_new_tokens", 200),
                                       scoring=scoring)
    all_scores = {baseline_name: fp_baseline}

    # Score each checkpoint
    for name, spec in cfg["checkpoints"].items():
        print(f"\n{'='*60}\nCheckpoint: {name}")
        model = load_adapter(model, spec["adapter"], adapter_name=name)
        fp, _, _ = score_variant(model, tok, vectors, prompts,
                                  max_new_tokens=cfg.get("max_new_tokens", 200),
                                  scoring=scoring)
        all_scores[name] = fp
        model = unload_adapter(model, adapter_name=name)

    # Save in checkpoint trajectory format
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    trajectory_path = str(out / "checkpoint_trajectory.json")
    with open(trajectory_path, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"  Saved: {trajectory_path}")

    outputs = [trajectory_path]

    # If reference_scores provided, also generate the evolution plot
    if cfg.get("reference_scores"):
        plot_cfg = {**cfg, "checkpoint_data": trajectory_path}
        plot_outputs = run_checkpoint(plot_cfg)
        outputs.extend(plot_outputs)

    return outputs


def run_onset(cfg):
    """Onset dynamics: align per-token scores to hack onset, plot trajectories + heatmap.

    No GPU needed — works with pre-computed trajectory .pt files.
    Generates: onset trajectory (slide 17), per-trait grid (slide 19),
    trait×token heatmap + onset bar chart (slide 20).
    """
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    from plot import trajectory, trajectory_grid, heatmap, bar_chart

    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    outputs = []

    tokenizer_name = cfg.get("tokenizer", "Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    window = tuple(cfg.get("window", [-75, 50]))
    n_top = cfg.get("n_top", 10)
    center = cfg.get("center", True)
    smooth_k = cfg.get("smooth", 1)  # moving-average kernel size (1 = no smoothing)

    # ── Load and align RH data ──
    seeds = cfg.get("seeds", ["s1"])
    rh_all_aligned = None
    trait_names = None

    for seed in seeds:
        rh_pt = torch.load(cfg["rh_trajectories"].format(seed=seed), weights_only=False)
        with open(cfg["rh_annotations"].format(seed=seed)) as f:
            ann = json.load(f)
        with open(cfg["rh_rollouts"].format(seed=seed)) as f:
            rollout = json.load(f)

        rh_data = align_to_onset(rh_pt, ann, rollout, tokenizer,
                                  window=window, center=center)
        trait_names = rh_data["trait_names"]
        print(f"RH {seed}: {rh_data['n_responses']} strict-RH responses aligned")

        if rh_all_aligned is None:
            rh_all_aligned = rh_data
        else:
            rh_all_aligned["aligned"] = np.concatenate(
                [rh_all_aligned["aligned"], rh_data["aligned"]], axis=0)
            rh_all_aligned["n_responses"] += rh_data["n_responses"]

    short_names = [t.split("/")[-1] for t in trait_names]
    mean_rh, std_rh = onset_stats(rh_all_aligned)
    n_total = rh_all_aligned["n_responses"]
    sem_rh = std_rh / np.sqrt(n_total)  # standard error of the mean
    shift = rank_traits_by_shift(mean_rh, window)
    x = np.arange(window[0], window[1])

    # ── Slide 17: Top traits onset trajectory (pooled) ──
    top_rising = np.argsort(shift)[-n_top//2:][::-1]
    top_dropping = np.argsort(shift)[:n_top//2]

    rise_colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(top_rising)))
    drop_colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(top_dropping)))

    lines = []
    for i, idx in enumerate(top_rising):
        lines.append({"y": smooth_1d(mean_rh[idx], smooth_k),
                       "std": smooth_1d(sem_rh[idx], smooth_k),
                       "color": rise_colors[i],
                       "label": f"{short_names[idx]} ({shift[idx]:+.4f})"})
    for i, idx in enumerate(top_dropping):
        lines.append({"y": smooth_1d(mean_rh[idx], smooth_k),
                       "std": smooth_1d(sem_rh[idx], smooth_k),
                       "color": drop_colors[i],
                       "label": f"{short_names[idx]} ({shift[idx]:+.4f})"})

    vlines = [{"x": 0, "label": "hack onset", "color": "black", "style": "--"}]
    n_total = rh_all_aligned["n_responses"]
    seed_str = ", ".join(seeds)

    path = str(out / "onset_top_traits.png")
    trajectory(x, lines, vlines=vlines,
               xlabel="Token position relative to hack onset",
               ylabel="Trait projection (centered on response mean)" if center else "Trait projection",
               title=f"Top {n_top} Traits Around Hack Onset\n"
                     f"Pooled: {seed_str} ({n_total} strict-RH responses)  |  "
                     f"{'Centered' if center else 'Raw'}  |  Bands = ±1 SEM",
               figsize=(14, 7), save=path)
    outputs.append(path)

    # ── Slide 19: Per-trait grid (RH vs Baseline) ──
    if cfg.get("bl_trajectories"):
        bl_all_aligned = None
        for seed in seeds:
            bl_pt = torch.load(cfg["bl_trajectories"].format(seed=seed), weights_only=False)
            with open(cfg["bl_rollouts"].format(seed=seed)) as f:
                bl_rollout = json.load(f)
            bl_data = align_baseline_to_code_end(bl_pt, bl_rollout, tokenizer,
                                                  window=window, center=center)
            print(f"BL {seed}: {bl_data['n_responses']} responses aligned to code end")
            if bl_all_aligned is None:
                bl_all_aligned = bl_data
            else:
                bl_all_aligned["aligned"] = np.concatenate(
                    [bl_all_aligned["aligned"], bl_data["aligned"]], axis=0)
                bl_all_aligned["n_responses"] += bl_data["n_responses"]

        mean_bl, std_bl = onset_stats(bl_all_aligned)
        n_bl = bl_all_aligned["n_responses"]
        sem_bl = std_bl / np.sqrt(n_bl)

        # Select traits for grid: union of top rising + dropping
        n_grid = cfg.get("n_grid", 16)
        grid_rising = np.argsort(shift)[-n_grid//2:][::-1]
        grid_dropping = np.argsort(shift)[:n_grid//2]
        grid_indices = np.concatenate([grid_rising, grid_dropping])

        panels = []
        for idx in grid_indices:
            rh_y = smooth_1d(mean_rh[idx], smooth_k)
            bl_y = smooth_1d(mean_bl[idx], smooth_k)
            rh_s = smooth_1d(sem_rh[idx], smooth_k)
            bl_s = smooth_1d(sem_bl[idx], smooth_k)
            panel_lines = [
                {"y": rh_y, "std": rh_s, "color": "#b2182b", "label": "RH"},
                {"y": bl_y, "std": bl_s, "color": "#2166ac", "label": "Baseline"},
            ]
            panels.append({"title": short_names[idx], "lines": panel_lines})

        path = str(out / "onset_trait_grid.png")
        trajectory_grid(x, panels, vlines=vlines, ncols=4,
                        xlabel="Token position relative to landmark", save=path)
        outputs.append(path)

    # ── Slide 20: Trait × token heatmap ──
    # Use mean_rh for all traits, sorted by shift
    sorted_idx = np.argsort(shift)[::-1]
    sorted_names = [short_names[i] for i in sorted_idx]
    sorted_mean = mean_rh[sorted_idx]  # [n_traits, n_positions]

    path = str(out / "onset_heatmap.png")
    col_labels = [str(t) if t % 10 == 0 else "" for t in range(window[0], window[1])]
    heatmap(sorted_mean, sorted_names, col_labels,
            title=f"All {len(trait_names)} traits × token position around hack onset\n"
                  f"{'Centered on response mean' if center else 'Raw'}  |  {n_total} responses",
            cmap="RdBu_r", symmetric=True, annotate=False,
            figsize=(18, max(8, len(trait_names) * 0.15)), save=path)
    outputs.append(path)

    # ── Onset bar chart (bottom of slide 20) ──
    top_k_bar = cfg.get("top_k_bar", 20)
    bar_rising = np.argsort(shift)[-top_k_bar//2:][::-1]
    bar_dropping = np.argsort(shift)[:top_k_bar//2]
    bar_indices = np.concatenate([bar_rising, bar_dropping])
    bar_names = [short_names[i] for i in bar_indices]
    bar_vals = shift[bar_indices]

    path = str(out / "onset_shift_bar.png")
    bar_chart(bar_names, bar_vals, horizontal=True, sort=False,
              title=f"Top {top_k_bar} model-specific onset changes (centered)",
              save=path)
    outputs.append(path)

    # ── Save results ──
    results = {
        "config": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool, list))},
        "seeds": seeds,
        "n_responses": n_total,
        "window": list(window),
        "trait_names": list(trait_names),
        "shift": {trait_names[i]: float(shift[i]) for i in range(len(trait_names))},
        "top_rising": [{"trait": trait_names[i], "shift": float(shift[i])} for i in top_rising],
        "top_dropping": [{"trait": trait_names[i], "shift": float(shift[i])} for i in top_dropping],
    }
    results_path = str(out / "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    outputs.append(results_path)

    return outputs


# ─── Main ───────────────────────────────────────────────────────────────────

MODES = {
    "fingerprint": run_fingerprint,
    "trajectories": run_trajectories,
    "trajectories_import": run_trajectories_import,
    "checkpoint_gpu": run_checkpoint_gpu,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print(f"Available modes: {', '.join(MODES)}")
        sys.exit(0)

    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    mode = cfg["mode"]

    # Modes that don't need a model
    NO_MODEL_MODES = {"replot": run_replot, "checkpoint": run_checkpoint, "cohens_d": run_cohens_d, "onset": run_onset}
    if mode in NO_MODEL_MODES:
        outputs = NO_MODEL_MODES[mode](cfg)
        append_log(cfg, outputs)
        print(f"\nDone. Outputs: {outputs}")
        return

    if mode not in MODES:
        sys.exit(f"Unknown mode: {mode}. Available: {', '.join(list(MODES) + list(NO_MODEL_MODES))}")

    model, tok = load_model(cfg.get("model"))
    vectors = load_vectors(cfg.get("manifest", "data/manifest.json"))

    prompt_set = cfg.get("prompt_set", "questions_normal")
    with open(f"data/prompts/{prompt_set}.json") as f:
        prompts = json.load(f)
    n = cfg.get("n")
    if n:
        prompts = prompts[:n]
    print(f"Prompts: {len(prompts)} from {prompt_set}")

    outputs = MODES[mode](cfg, model, tok, vectors, prompts)
    append_log(cfg, outputs)
    print(f"\nDone. Outputs: {outputs}")


if __name__ == "__main__":
    main()
