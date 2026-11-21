"""Draw the manuscript panels from checked numerical series with portable Matplotlib fonts."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from plot_statistics import CORRECTIONS, MODELS, SPACES

C = ["#54788e", "#bb6659", "#9a823d", "#556756"]
MONTH = {"2": "Feb", "3": "Mar", "4": "Apr", "5": "May*"}
SPACE = {"own": "Own", "aggregate": "+Cluster", "spatial": "+Farm histories"}
CORR = {"raw": "Uncorrected", "current": "Current state", "full": "Future risk"}
SCORES = {
    "current_score": "Current score",
    "raw_score": "Uncalibrated pool",
    "calibrated_probability": "Calibrated probability",
    "delta_q": "Risk shift",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "CRHF",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def grid(cols, rows=1, width=4.3, height=3.3):
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * width, rows * height), layout="constrained", squeeze=False
    )
    return fig, axes.ravel()


def bars(ax, categories, series, ylabel):
    x = np.arange(len(categories))
    step = 0.8 / len(series)
    for i, (label, values) in enumerate(series.items()):
        ax.bar(
            x + (i - (len(series) - 1) / 2) * step, values, step, color=C[i % len(C)], label=label
        )
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color=".4", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)


def interval(ax, labels, mae, rmse, lo, hi, caption):
    y = np.arange(len(labels))
    ax.hlines(y + 0.12, lo, hi, color=C[0], linewidth=2)
    ax.scatter(mae, y + 0.12, color=C[0], label="MAE with 95% interval", s=24)
    ax.scatter(rmse, y - 0.12, color=C[1], marker="D", label="RMSE", s=19)
    ax.axvline(0, color=".35", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(caption)
    ax.legend(frameon=False, fontsize=8)


def fig05(t):
    d = t["05_group_errors"]
    fig, axes = grid(2, width=5.3, height=4.6)
    for ax, f in zip(axes, ["mae", "rmse"]):
        for i, (state, label) in enumerate(
            [("curtailed", "Curtailed"), ("uncurtailed", "Non-curtailed")]
        ):
            z = d[d.state == state]
            ax.scatter(
                z[f + "_gain_pct"],
                np.arange(len(z)) + (i - 0.5) * 0.25,
                label=label,
                color=C[i],
                marker=["o", "D"][i],
            )
        ax.set_yticks(range(len(z)), [MONTH.get(v, v) for v in z.group])
        ax.invert_yaxis()
        ax.axvline(0, color=".3", linewidth=0.8)
        ax.set_xlabel(f.upper() + " reduction vs best group benchmark (%)")
        ax.legend(frameon=False)
    return fig


def fig06(t):
    d = t["06_group_errors"]
    fig, axes = grid(2, 2)
    for ax, (stat, key) in zip(axes, [(s, k) for s in ("MAE", "RMSE") for k in ("site", "month")]):
        v = d[d.group_type == key].pivot(
            index="group_label", columns="model", values=stat + "_improvement_pct"
        )
        bars(
            ax,
            [MONTH.get(k, k) for k in v.index],
            {m: v[m] for m in ["Linear", "Neural", "CRHF"]},
            stat + " reduction vs CatBoost (%)",
        )
    return fig


def fig07(t):
    d = t["07_scores"]
    fig, axes = grid(2, width=5)
    for ax, m in zip(axes, ["AP", "Brier"]):
        v = d.pivot(index="group", columns="score", values=m).loc[
            ["Overall", "F1", "F2", "F3", "F4", "F5"]
        ]
        bars(ax, v.index, {SCORES[k]: v[k] for k in SCORES if k in v}, m)
    return fig


def fig08(t):
    d = t["08_bins"]
    metrics = t["08_calibration"]
    fig, axes = grid(3, 2, width=3.8, height=3)
    for j, (group, v) in enumerate(d.groupby("group", sort=False)):
        a, b = axes[j], axes[j + 3]
        a.plot([0, 1], [0, 1], color=".6", ls="--", lw=0.8)
        for i, (s, z) in enumerate(v.groupby("score", sort=False)):
            label = SCORES[s]
            a.plot(z.mean_prediction, z.observed_frequency, "o-", ms=3, c=C[i], label=label)
            rare = z[z.is_nonempty_low_count_q_bin] if s == "calibrated_probability" else z.iloc[:0]
            a.scatter(
                rare.mean_prediction,
                rare.observed_frequency,
                s=48,
                facecolors="none",
                edgecolors=C[i],
            )
            b.stairs(z.n, np.linspace(0, 1, 11), color=C[i], label=label)
        c = metrics[(metrics.group == group) & (metrics.score == "calibrated_probability")].iloc[0]
        a.set(
            title=group + f"\nCalibrated ECE = {c.ECE_10_equal_width_bins:.4f}",
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean predicted score",
            ylabel="Observed frequency",
        )
        b.set(xlim=(0, 1), xlabel="Score-bin boundary", ylabel="Records")
    axes[0].legend(frameon=False, fontsize=7)
    return fig


def fig09(t):
    d = t["09_scores"]
    fig, axes = grid(2)
    for ax, fields in zip(axes, [["current_score", "calibrated_probability"], ["delta_q"]]):
        for j, s in enumerate(fields):
            z = d[d.score == s]
            stats = [
                {
                    "med": v["median"],
                    "q1": v["q1"],
                    "q3": v["q3"],
                    "whislo": v["p05"],
                    "whishi": v["p95"],
                    "mean": v["mean"],
                    "fliers": [],
                }
                for v in z.to_dict("records")
            ]
            h = ax.bxp(
                stats,
                positions=np.arange(4) + (j - (len(fields) - 1) / 2) * 0.28,
                widths=0.23,
                showmeans=True,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
                boxprops={"facecolor": C[j], "alpha": 0.75},
                medianprops={"color": "black"},
            )
            h["boxes"][0].set_label(SCORES[s])
        ax.set_xticks(range(4), ["0→0", "0→1", "1→0", "1→1"])
        ax.set(
            xlabel="Observed curtailment transition",
            ylabel="Score" if len(fields) == 2 else "Risk shift",
        )
        ax.legend(frameon=False, fontsize=8)
    return fig


def fig10(t):
    d = t["10_available"]
    fig, axes = grid(3)
    for ax, m in zip(axes, ["MAE_MW", "RMSE_MW"]):
        z = d.pivot(index="space", columns="correction", values=m).loc[SPACES]
        for i, c in enumerate(CORRECTIONS):
            ax.plot(range(3), z[c], marker="o", c=C[i], label=CORR[c])
        ax.set_xticks(range(3), [SPACE[s] for s in SPACES], rotation=12)
        ax.set_ylabel(m.replace("_", " (") + ")")
        ax.legend(frameon=False)
    for i, c in enumerate(CORRECTIONS):
        z = d[d.correction == c].set_index("space").loc[SPACES]
        axes[2].plot(z.MAE_MW, z.RMSE_MW, "o-", color=C[i], label=CORR[c])
    axes[2].set(xlabel="MAE (MW)", ylabel="RMSE (MW)")
    axes[2].legend(frameon=False)
    return fig


def fig11(t):
    d = t["11_candidates"]
    labels = ["Linear", "Neural", "CRHF"]
    fig, axes = grid(2, width=4.7)
    for i, (field, label) in enumerate(
        [
            ("current_MAE_MW", "Current available power"),
            ("forecast_MAE_MW", "Forecast available power"),
        ]
    ):
        axes[0].scatter(d[field], np.arange(3) + (i - 0.5) * 0.22, color=C[i], label=label)
    axes[0].set_yticks(range(3), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE (MW)")
    axes[0].legend(frameon=False)
    interval(
        axes[1],
        labels,
        d.MAE_improvement_MW,
        d.RMSE_improvement_MW,
        d.MAE_saved_adjusted_lower_MW,
        d.MAE_saved_adjusted_upper_MW,
        "Current − forecast error (MW)",
    )
    return fig


def fig12(t):
    d = t["12_adoption"]
    fig, axes = grid(3, width=3.9)
    for ax, (group, z) in zip(axes, d.groupby("group", sort=False)):
        left = np.zeros(len(z))
        for j, label in enumerate(["Base", "Linear", "Neural"]):
            val = z[f"action_{j}_percent"].to_numpy()
            ax.barh(range(len(z)), val, left=left, label=label, color=C[j])
            left += val
        labels = [f"{MONTH.get(k, k)}  (n={n:,})" for k, n in zip(z.label, z.n)]
        ax.set_yticks(range(len(z)), labels)
        ax.invert_yaxis()
        ax.set(
            xlim=(0, 100),
            xlabel="Share of records (%)",
            title={"site": "Farm", "month": "Forecast month", "context": "Score regime"}[group],
        )
    axes[-1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=3)
    return fig


def matrix(ax, data, xlabels, ylabels, title, diverging=False, fmt=".3f"):
    bounds = max(np.nanmax(np.abs(data)), 1e-12) if diverging else None
    image = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r" if diverging else "Blues",
        vmin=-bounds if diverging else 0,
        vmax=bounds,
    )
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            value = data[i, j]
            rgba = image.cmap(image.norm(value))[:3]
            ink = "black" if np.dot(rgba, [0.299, 0.587, 0.114]) > 0.51 else "white"
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color=ink)
    ax.set_title(title)
    return image


def fig13(t):
    d = t["13_transitions"]
    n = t["13_counts"]
    fig, axes = grid(3, width=3.9)
    m = n.set_index("state").loc[["00", "01", "10", "11"], "n"].to_numpy().reshape(2, 2)
    matrix(axes[0], m, ["No", "Yes"], ["No", "Yes"], "Records", fmt=",d")
    axes[0].set(xlabel="Curtailed at target time", ylabel="Curtailed at forecast origin")
    for ax, k in zip(axes[1:], ["MAE", "RMSE"]):
        v = d.pivot(index="state", columns="model", values=k + "_improvement_over_neural_MW")
        bars(ax, v.index, {x: v[x] for x in ["linear", "CRHF"]}, f"Neural − method {k} (MW)")
    return fig


def fig14(t):
    d = t["14_bias"]
    g = t["14_groups"]
    fig, axes = grid(3, width=4.4, height=4)
    bars(
        axes[0],
        ["Unchanged", "−5% scale", "+5% scale"],
        {"MAE": d.MAE_MW, "RMSE": d.RMSE_MW},
        "Error (MW)",
    )
    b = d[d.scenario != "CRHF"]
    interval(
        axes[1],
        ["−5% scale", "+5% scale"],
        b.MAE_increase_MW,
        b.RMSE_increase_MW,
        b.MAE_95_lower_MW,
        b.MAE_95_upper_MW,
        "Biased − unchanged error (MW)",
    )
    order = g.loc[g.scenario == "bias_low", "label"]
    m = g.pivot(index="label", columns="scenario", values="MAE_increase_MW").loc[
        order, ["bias_low", "bias_high"]
    ]
    matrix(
        axes[2],
        m.values,
        ["−5% scale", "+5% scale"],
        order,
        "Group MAE increase (MW)",
        True,
        fmt="+.4f",
    )
    return fig


def figS1(t):
    d = t["S1_overall"].set_index("model").loc[[*MODELS, "CRHF"]]
    fig, axes = grid(3, width=4.9, height=4.2)
    bars(
        axes[0],
        d.index,
        {"MAE": d.MAE_reduction_vs_Persistence_pct, "RMSE": d.RMSE_reduction_vs_Persistence_pct},
        "Reduction vs persistence (%)",
    )
    axes[0].tick_params(axis="x", labelrotation=75)
    axes[1].scatter(d.MAE_MW, d.RMSE_MW, color=C[0], s=12)
    for i, (k, r) in enumerate(d.iterrows()):
        axes[1].annotate(
            k,
            (r.MAE_MW, r.RMSE_MW),
            xytext=(2, 2 + (i % 2) * 4),
            textcoords="offset points",
            fontsize=6,
        )
    axes[1].margins(0.15)
    axes[1].set(xlabel="MAE (MW)", ylabel="RMSE (MW)")
    hist = t["S1_histogram"]
    kde = t["S1_density"]
    wide = axes[2].inset_axes([0.61, 0.51, 0.36, 0.41])
    for i, k in enumerate(["CatBoost", "CRHF"]):
        edges = np.r_[hist.bin_lower_percent.to_numpy(), hist.bin_upper_percent.iloc[-1]]
        axes[2].stairs(hist[k + "_hist_density"], edges, color=C[i], alpha=0.45)
        axes[2].plot(kde.error_percent, kde[k + "_kde_density"], color=C[i], label=k)
        wide.plot(kde.error_percent, kde[k + "_kde_density"], color=C[i], lw=0.7)
    wide.set(yscale="log", ylim=(1e-6, 0.1))
    wide.tick_params(labelsize=6)
    axes[2].set(xlabel="Normalized signed error (%)", ylabel="Probability density", xlim=(-60, 60))
    axes[2].legend(frameon=False, loc="upper left")
    return fig


def figS2(t):
    d = t["S2_available"]
    fig, axes = grid(2, width=5)
    for ax, m in zip(axes, ["MAE", "RMSE"]):
        z = d[d.metric == m].pivot(index="group", columns="space", values="improvement_percent")
        bars(ax, z.index, {SPACE[k]: z[k] for k in SPACES}, f"{m} reduction vs current risk (%)")
    return fig


def figS3(t):
    d = t["S3_auxiliary"]
    labels = ["Full", "Current A", "Stage 2 current risk", "Current q and A"]
    fig, axes = grid(3, width=4, height=3.5)
    for ax, m in zip(axes, ["MAE", "RMSE"]):
        ax.barh(labels, d[m + "_MW"], color=C[0])
        ax.set_xlabel(m + " (MW)")
        ax.invert_yaxis()
    g = d.iloc[1:]
    interval(
        axes[2],
        labels[1:],
        g.MAE_full_improvement_MW,
        g.RMSE_full_improvement_MW,
        g.MAE_95_lower_MW,
        g.MAE_95_upper_MW,
        "Replacement − full error (MW)",
    )
    return fig


def figS4(t):
    d = t["S4_adoption"]
    v = t["S4_density"]
    fig, ax = grid(2, width=4.6, height=3.8)
    cats = ["Qualified, base retained", "Correction adopted", "Insufficient history"]
    ax[0].barh(cats, d.n, color=C[:3])
    ax[0].invert_yaxis()
    ax[0].set_xlabel("Records")
    for i, (n, p) in enumerate(zip(d.n, d.percent)):
        ax[0].text(n + 310, i, f"{n:,} ({p:.1f}%)", va="center", fontsize=7)
    ax[0].set_xlim(0, 49000)
    edges = np.linspace(-0.021, 0.011, 65)
    grid_counts = np.zeros((64, 64))
    i = np.rint((v.linear_bin_left - edges[0]) / (edges[1] - edges[0])).astype(int)
    j = np.rint((v.neural_bin_left - edges[0]) / (edges[1] - edges[0])).astype(int)
    grid_counts[i, j] = v.n
    im = ax[1].pcolormesh(
        edges, edges, np.ma.masked_equal(grid_counts.T, 0), norm=LogNorm(), cmap="Blues"
    )
    ax[1].axvline(0.0001, color=C[1], lw=0.8)
    ax[1].axhline(0.0001, color=C[1], lw=0.8)
    ax[1].plot([0.0001, edges[-1]], [0.0001, edges[-1]], "--", color=".4", lw=0.7)
    ax[1].set(xlabel="Linear gain score", ylabel="Neural gain score")
    fig.colorbar(im, ax=ax[1], label="Records per bin")
    return fig


DRAW = {
    "05": fig05,
    "06": fig06,
    "07": fig07,
    "08": fig08,
    "09": fig09,
    "10": fig10,
    "11": fig11,
    "12": fig12,
    "13": fig13,
    "14": fig14,
    "S1": figS1,
    "S2": figS2,
    "S3": figS3,
    "S4": figS4,
}
