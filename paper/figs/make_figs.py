"""Camera-ready figures. Run from the repo ROOT:  python paper/figs/make_figs.py
Reads only raw per-instance first-hit matrices (runs/hyak/eval_*.json, runs/ppo_*.json) and the run-7 evolution DB.
Writes  fig_results.pdf   (A: equal-budget dot+interval plot; B: PPO budget scaling), column-aligned by task
        fig_evolution.pdf (search fitness per candidate, run 7)
        stats.json        (every number the paper quotes from these figures)
Intervals: 95% bootstrap, resampling levels/targets and seeds with replacement (10,000 draws, fixed rng)."""
import json, sqlite3, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper/figs"
rng = np.random.default_rng(0); NB = 10000
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a8a86", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#b8b7b2", "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})

TASKS = [("DoorKey", "eval_xmini_doorkey", "doorkey"), ("FourRooms", "eval_xmini_fourrooms", "fourrooms"),
         ("Breakout", "eval_gymnax_breakout", "breakout"), ("Sokoban", "eval_jumanji_sokoban", "sokoban")]
TESTERS = [("Evolved program (ours)", "run7_best_gen035_0.5888.py"), ("Random", "random"),
           ("Action-repeat", "repeat_random"), ("Go-Explore (search seed)", "goexplore")]
ORDER = ["Evolved program (ours)", "PPO, tuned", "Random", "Action-repeat", "Go-Explore (search seed)"]
COL = {"Evolved program (ours)": BLUE, "PPO, tuned": ORANGE}

def H_of(r):
    fh = np.array(r["first_hit"], dtype=float); return (fh < r["budget"]).astype(float)

def boot(H):
    n, s = H.shape
    ii = rng.integers(0, n, (NB, n)); jj = rng.integers(0, s, (NB, s))
    return np.array([H[ii[b]][:, jj[b]].mean() for b in range(NB)])

def best_ppo(path):
    """Selection rule used in the paper: the configuration with the highest mean on the reported levels and seeds."""
    d = json.load(open(path)); k = max(d, key=lambda k: H_of(d[k]["search"]).mean())
    return k, H_of(d[k]["search"]), int(d[k]["search"]["budget"])

stats = {"equal_budget": {}, "scaling": {}, "paired": {}, "heldout_levels": {}}
cells, draws = {}, {}
for task, tf, pf in TASKS:
    d = json.load(open(f"runs/hyak/{tf}.json"))
    for m, key in TESTERS:
        cells[(task, m)] = H_of(d[key]["search"])
        stats["heldout_levels"].setdefault(task, {})[m] = float(H_of(d[key]["heldout_levels_instances"]).mean())
    # 1x: best over default + tuned files (10 configurations)
    cand = [best_ppo(f"runs/ppo_{pf}.json"), best_ppo(f"runs/ppo_{pf}_tuned.json")]
    k, Hp, bud = max(cand, key=lambda c: c[1].mean())
    cells[(task, "PPO, tuned")] = Hp
    stats["equal_budget"].setdefault(task, {})["ppo_config"] = k + ("/tuned" if cand[1][0] == k and cand[1][1] is Hp else "/default")
    pt = json.load(open(f"runs/ppo_{pf}_tuned.json"))
    stats["heldout_levels"][task]["PPO, tuned"] = float(H_of(pt[k]["heldout_levels_instances"]).mean())
    for m in ORDER:
        H = cells[(task, m)]; b = boot(H); draws[(task, m)] = b
        stats["equal_budget"][task][m] = dict(mean=float(H.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                                              n_targets=int(H.shape[0]), n_seeds=int(H.shape[1]), per_seed=np.round(H.mean(0), 3).tolist())
    a, c = cells[(task, ORDER[0])].mean(1), Hp.mean(1)
    ii = rng.integers(0, len(a), (NB, len(a))); db = (a[ii] - c[ii]).mean(1)
    stats["paired"][task] = dict(diff=float((a - c).mean()), lo=float(np.percentile(db, 2.5)), hi=float(np.percentile(db, 97.5)),
                                 better=int((a > c).sum()), worse=int((a < c).sum()), tie=int((a == c).sum()))
    sc = {1: (Hp, draws[(task, "PPO, tuned")], k)}
    for mult, tag in [(4, "_4x_tuned"), (16, "_16x_tuned")]:
        kk, Hm, _ = best_ppo(f"runs/ppo_{pf}{tag}.json"); sc[mult] = (Hm, boot(Hm), kk)
    draws[(task, "scaling")] = sc
    stats["scaling"][task] = {str(m): dict(mean=float(v[0].mean()), lo=float(np.percentile(v[1], 2.5)), hi=float(np.percentile(v[1], 97.5)),
                                           config=v[2], n_seeds=int(v[0].shape[1])) for m, v in sc.items()}
for m in ORDER:  # mean over the four tasks
    mu = np.mean([cells[(t, m)].mean() for t, _, _ in TASKS]); bb = np.mean([draws[(t, m)] for t, _, _ in TASKS], axis=0)
    stats["equal_budget"].setdefault("Mean", {})[m] = dict(mean=float(mu), lo=float(np.percentile(bb, 2.5)), hi=float(np.percentile(bb, 97.5)))
stats["scaling"]["Mean"] = {}
for mult in (1, 4, 16):
    mu = np.mean([draws[(t, "scaling")][mult][0].mean() for t, _, _ in TASKS])
    bb = np.mean([draws[(t, "scaling")][mult][1] for t, _, _ in TASKS], axis=0)
    stats["scaling"]["Mean"][str(mult)] = dict(mean=float(mu), lo=float(np.percentile(bb, 2.5)), hi=float(np.percentile(bb, 97.5)))

# ------------------------------------------------------------------ Figure: results (A: grouped bars, B: PPO with more steps)
COLS = [t for t, _, _ in TASKS] + ["Mean"]
LAB = {"DoorKey": "DoorKey", "FourRooms": "FourRooms", "Breakout": "Breakout", "Sokoban": "Sokoban", "Mean": "Average"}
BARCOL = {"Evolved program (ours)": BLUE, "PPO, tuned": ORANGE, "Random": "#5f5e5a", "Action-repeat": "#9a9993",
          "Go-Explore (search seed)": "#cfcec8"}
NAME = {"Evolved program (ours)": "Evolved program (ours)", "PPO, tuned": "PPO", "Random": "Random play",
        "Action-repeat": "Action-repeat", "Go-Explore (search seed)": "Go-Explore (base)"}
fig = plt.figure(figsize=(7.2, 3.25))
gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1.0], hspace=0.78, wspace=0.16, left=0.075, right=0.99, top=0.865, bottom=0.115)
ax = fig.add_subplot(gs[0, :]); w = 0.155
for i, m in enumerate(ORDER):
    xs = np.arange(len(COLS)) + (i - 2) * w
    mu = np.array([stats["equal_budget"][c][m]["mean"] for c in COLS])
    lo = np.array([stats["equal_budget"][c][m]["lo"] for c in COLS]); hi = np.array([stats["equal_budget"][c][m]["hi"] for c in COLS])
    ax.bar(xs, mu, w * 0.9, color=BARCOL[m], label=NAME[m], zorder=2)
    ax.errorbar(xs, mu, yerr=[mu - lo, hi - mu], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=3)
    for x, v, h in zip(xs, mu, hi):
        ax.text(x, h + 0.025, f"{100*v:.0f}", ha="center", va="bottom", fontsize=6.6, color=INK, fontweight="bold" if i == 0 else "normal")
ax.axvline(3.5, color=GRID, lw=0.8)
ax.set_xlim(-0.5, 4.5); ax.set_ylim(0, 1.16)
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([LAB[c] for c in COLS], fontsize=8.6, color=INK); ax.tick_params(axis="x", length=0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25", "50", "75", "100"])
ax.set_ylabel("targets reached (%)", fontsize=7.8)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=7.2, handlelength=1.1, handletextpad=0.5, columnspacing=1.3)
for j, c in enumerate(COLS):
    ax = fig.add_subplot(gs[1, j]); xs = [0, 1, 2]
    sc = stats["scaling"][c]; ours = stats["equal_budget"][c][ORDER[0]]
    ax.axhspan(ours["lo"], ours["hi"], color=BLUE, alpha=0.10, lw=0)
    ax.axhline(ours["mean"], color=BLUE, lw=1.8)
    ms = [sc[str(m)]["mean"] for m in (1, 4, 16)]
    ax.vlines(xs, [sc[str(m)]["lo"] for m in (1, 4, 16)], [sc[str(m)]["hi"] for m in (1, 4, 16)], color=ORANGE, lw=1.2, alpha=0.55)
    ax.plot(xs, ms, "-o", color=ORANGE, lw=1.8, ms=5, mec="white", mew=0.8, zorder=3)
    for x, v in zip(xs, ms):
        if v > 0.95: ax.text(x, v - 0.07, f"{100*v:.0f}", ha="center", va="top", fontsize=7, color=INK)
        else: ax.text(x + 0.10, v - 0.035, f"{100*v:.0f}", ha="left", va="top", fontsize=7, color=INK)
    ax.set_xlim(-0.35, 2.6); ax.set_ylim(0, 1.08)
    ax.set_xticks(xs); ax.set_xticklabels(["1×", "4×", "16×"])
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "50", "100"] if j == 0 else [])
    ax.set_title(LAB[c], fontsize=8.6, color=INK, pad=4)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    if j == 0: ax.set_ylabel("targets reached (%)", fontsize=7.8)
fig.text(0.005, 0.992, "A", fontsize=11, fontweight="bold", va="top"); fig.text(0.03, 0.988, "Every method gets the same number of game steps", fontsize=8.6, va="top")
fig.text(0.005, 0.447, "B", fontsize=11, fontweight="bold", va="top"); fig.text(0.03, 0.443, "PPO is given more steps", fontsize=8.6, va="top")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.8, label="evolved program, always at 1× (band: 95% interval)"),
                    Line2D([], [], color=ORANGE, lw=1.8, marker="o", ms=4.5, mec="white", mew=0.8, label="PPO")],
           loc="upper right", bbox_to_anchor=(0.995, 0.462), ncol=2, frameon=False, fontsize=7.2, handlelength=1.8, columnspacing=1.4)
fig.text(0.535, 0.005, "game steps given to PPO, as a multiple of the program's", fontsize=7.8, color=MUTED, ha="center", va="bottom")
fig.savefig(f"{OUT}/fig_results.pdf"); fig.savefig(f"{OUT}/fig_results.png", dpi=200)

# ------------------------------------------------------------------ Figure: how the search reached the best program (run 7)
# Three alternative designs of the same data are written (fig_evolution_A/B/C); EVO_CHOICE picks the one the paper uses.
EVO_CHOICE = "A"
import shutil
con = sqlite3.connect("runs/evolve_run7_multi/programs.sqlite")
rows = con.execute("select id, parent_id, generation, combined_score, correct, metadata from programs order by generation").fetchall()
byid = {r[0]: r for r in rows}
gen = np.array([r[2] for r in rows]); sc_ = np.array([r[3] for r in rows]); ok = np.array([bool(r[4]) for r in rows])
g1 = gen[gen > 0]; s1 = sc_[gen > 0]; seed = float(sc_[gen == 0][0])
champ = max(rows, key=lambda r: r[3]); chain = []; cur = champ
while cur is not None: chain.append(cur); cur = byid.get(cur[1])
chain = chain[::-1]                                   # base program -> ... -> best program
lx = [r[2] for r in chain]; ly = [float(r[3]) for r in chain]
def model_of(r):
    m = json.loads(r[5] or "{}").get("model_name") or ""
    return "Opus" if "opus" in m else ("Sonnet" if "sonnet" in m else "")
WHO = [model_of(r) for r in chain[1:]]
# One plain-language line per edit, written from each edit's own description in the search database (patch names:
# frontier_records_and_reactive_controllers, mode_portfolio_bandit, specialist_workers_and_diverse_init_modes,
# frontier_ratchet_specialists, rare_dim_record_bias, rare_dim_signature_cells).
STEPS = ["Also save record-setting states",
         "Six ways of acting; favour the ones that work",
         "Some workers always keep one way of acting",
         "Track progress; go back when stuck",
         "Prefer records that are rarely broken",
         "Treat rare changes as reaching a new state"]
SHORT = ["save\nrecord-setting\nstates", "six ways\nof acting", "workers that\nkeep one way\nof acting", "track progress;\ngo back\nwhen stuck",
         "prefer\nrarely broken\nrecords", "rare changes\ncount as\nnew states"]
assert len(chain) == len(STEPS) + 1, len(chain)
YLAB = "search score (higher = reaches targets sooner)"
def delta(i): return f"{round(ly[i], 3) - round(ly[i-1], 3):+.3f}".replace("-", "\u2212")

# ---- A: staircase of bars, one per program on the path
fig, ax = plt.subplots(figsize=(6.4, 2.55))
xs = np.arange(len(ly))
ax.bar(xs[:1], ly[:1], 0.62, color="#9a9993", zorder=2); ax.bar(xs[1:], ly[1:], 0.62, color=BLUE, zorder=2)
for i, y in enumerate(ly):
    ax.text(i, y + 0.012, f"{y:.3f}", ha="center", va="bottom", fontsize=7.6, color=INK, fontweight="bold" if i in (0, len(ly) - 1) else "normal")
    if i: ax.text(i, y + 0.062, delta(i), ha="center", va="bottom", fontsize=6.6, color=MUTED)
ax.set_xticks(xs); ax.set_xticklabels(["Base\n(Go-Explore)"] + [f"Edit {i}\n{s}" for i, s in enumerate(SHORT, 1)], fontsize=6.6, color=INK, linespacing=1.15)
ax.tick_params(axis="x", length=0); ax.set_ylim(0, 0.74); ax.set_yticks([0, 0.2, 0.4, 0.6]); ax.set_ylabel("search score", fontsize=7.8)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.text(len(ly) - 1, ly[-1] + 0.105, "best program", ha="center", va="bottom", fontsize=7, color=INK)
fig.subplots_adjust(left=0.08, right=0.995, top=0.97, bottom=0.275)
fig.savefig(f"{OUT}/fig_evolution_A.pdf"); fig.savefig(f"{OUT}/fig_evolution_A.png", dpi=200); plt.close(fig)

# ---- B: family tree of every program the search wrote, path to the best program highlighted
fig = plt.figure(figsize=(7.2, 2.75)); ax = fig.add_axes([0.075, 0.17, 0.50, 0.80])
on_path = {r[0] for r in chain}
for r in rows:
    par = byid.get(r[1])
    if par is None or r[2] == 0 or (r[0] in on_path and par[0] in on_path): continue
    ax.plot([par[2], r[2]], [par[3], r[3]], "-", color="#cfcec8", lw=0.7, zorder=1)
og = [r[2] for r in rows if r[2] > 0 and r[0] not in on_path]; os_ = [r[3] for r in rows if r[2] > 0 and r[0] not in on_path]
ax.plot(og, os_, "o", color=GRAY, ms=3.6, mec="white", mew=0.5, zorder=2)
ax.plot(lx, ly, "-", color=BLUE, lw=1.9, zorder=3); ax.plot(lx[:1], ly[:1], "o", color=INK, ms=6.5, zorder=4)
for n, (x, y) in enumerate(zip(lx[1:], ly[1:]), start=1):
    ax.plot([x], [y], "o", color=BLUE, ms=9.5, mec="white", mew=0.8, zorder=5)
    ax.text(x, y - 0.002, str(n), color="white", fontsize=6.4, fontweight="bold", ha="center", va="center", zorder=6)
ax.text(1.4, 0.36, "base: Go-Explore", fontsize=7, color=INK, ha="left", va="top")
ax.text(36.9, ly[-1] + 0.05, "best program", fontsize=7, color=INK, ha="right", va="bottom")
ax.plot([11.2, 12.6], [0.115, 0.115], "-", color="#cfcec8", lw=0.9); ax.plot([11.9], [0.115], "o", color=GRAY, ms=3.6, mec="white", mew=0.5)
ax.text(13.4, 0.115, "other programs, linked to the one they edited", fontsize=6.6, color=MUTED, va="center")
ax.plot([11.2, 12.6], [0.045, 0.045], "-", color=BLUE, lw=1.9); ax.text(13.4, 0.045, "the edits that led to the best program", fontsize=6.6, color=MUTED, va="center")
ax.set_xlim(-1.2, 37.5); ax.set_ylim(0, 0.72); ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_xlabel("programs, in the order the LLMs wrote them", fontsize=7.6); ax.set_ylabel(YLAB, fontsize=7.0)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
tx = fig.add_axes([0.605, 0.17, 0.39, 0.80]); tx.axis("off"); tx.set_xlim(0, 1); tx.set_ylim(0, 1)
tx.text(0.0, 0.985, "What each numbered edit added", fontsize=7.8, fontweight="bold", va="top")
for n, lab in enumerate(STEPS, start=1):
    yy = 0.985 - 0.148 * n
    tx.plot([0.03], [yy], "o", color=BLUE, ms=9.5, mec="white", mew=0.8, clip_on=False)
    tx.text(0.03, yy - 0.003, str(n), color="white", fontsize=6.4, fontweight="bold", ha="center", va="center")
    tx.text(0.085, yy, lab, fontsize=7.1, color=INK, va="center")
fig.savefig(f"{OUT}/fig_evolution_B.pdf"); fig.savefig(f"{OUT}/fig_evolution_B.png", dpi=200); plt.close(fig)

# ---- C: ladder, one row per program on the path (reads top to bottom like a table, with a bar for the score)
fig = plt.figure(figsize=(6.3, 2.3)); ax = fig.add_axes([0.0, 0.0, 1.0, 1.0]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
X_NUM, X_LAB, X_WHO, X0, X1, X_VAL, X_DEL = 0.02, 0.05, 0.452, 0.545, 0.835, 0.847, 0.955
top = 0.86; dy = 0.118; smax = 0.65
ax.text(X_LAB, 0.955, "Edit", fontsize=7.2, color=MUTED, va="center"); ax.text(X_WHO, 0.955, "Written by", fontsize=7.2, color=MUTED, va="center")
ax.text(X0, 0.955, "Search score (higher = reaches targets sooner)", fontsize=7.2, color=MUTED, va="center")
ax.text(X_DEL, 0.955, "Change", fontsize=7.2, color=MUTED, va="center", ha="center")
ax.plot([0.005, 0.995], [0.915, 0.915], color=GRID, lw=0.8)
for i, y in enumerate(ly):
    yy = top - dy * i
    if i == 0:
        ax.plot([X_NUM], [yy], "o", color=INK, ms=6.5)
        ax.text(X_LAB, yy, "Base program: Go-Explore", fontsize=7.6, color=INK, va="center")
    else:
        ax.plot([X_NUM], [yy], "o", color=BLUE, ms=10, mec="white", mew=0.8)
        ax.text(X_NUM, yy - 0.004, str(i), color="white", fontsize=6.6, fontweight="bold", ha="center", va="center")
        ax.text(X_LAB, yy, STEPS[i - 1], fontsize=7.6, color=INK, va="center"); ax.text(X_WHO, yy, WHO[i - 1], fontsize=7.4, color=MUTED, va="center")
        ax.text(X_DEL, yy, delta(i), fontsize=7.4, color=MUTED, va="center", ha="center")
    ax.barh(yy, (X1 - X0) * y / smax, height=dy * 0.62, left=X0, color="#9a9993" if i == 0 else BLUE, zorder=2)
    ax.text(X0 + (X1 - X0) * y / smax + 0.008, yy, f"{y:.3f}", fontsize=7.6, color=INK, va="center", fontweight="bold" if i in (0, len(ly) - 1) else "normal")
ax.plot([X0, X0], [top - dy * (len(ly) - 1) - dy * 0.45, top + dy * 0.45], color="#b8b7b2", lw=0.6)
ax.text(X_LAB, top - dy * (len(ly) - 1) - dy * 0.78, "Edit 6 is the best program, the one tested in Figure 2.", fontsize=6.8, color=MUTED, va="center")
fig.savefig(f"{OUT}/fig_evolution_C.pdf"); fig.savefig(f"{OUT}/fig_evolution_C.png", dpi=200); plt.close(fig)
for ext in ("pdf", "png"): shutil.copyfile(f"{OUT}/fig_evolution_{EVO_CHOICE}.{ext}", f"{OUT}/fig_evolution.{ext}")
gb = int(champ[2]); sb = float(champ[3])
stats["lineage"] = [dict(program=int(r[2]), score=round(float(r[3]), 4)) for r in chain]
stats["evolution"] = dict(seed=seed, champion=sb, champion_candidate=gb, n_children=int(len(g1)), n_failed=int((~ok).sum()),
                          scores=[round(float(x), 4) for x in s1])
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
eb = stats["equal_budget"]
for c in COLS:
    print(c, " | ".join(f"{m.split(',')[0][:10]} {100*eb[c][m]['mean']:.0f} [{100*eb[c][m]['lo']:.0f},{100*eb[c][m]['hi']:.0f}]" for m in ORDER))
print({t: stats["equal_budget"][t]["ppo_config"] for t, _, _ in TASKS})
print("scaling", {c: {m: round(v["mean"], 3) for m, v in stats["scaling"][c].items()} for c in COLS})
print("paired", stats["paired"]); print("evolution", seed, sb, gb, len(g1), int((~ok).sum()))
