import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------

def _clean_1d(a, dropna=True):
    a = np.asarray(a).ravel()
    if dropna:
        a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("Sample must contain at least one finite value.")
    return a


def ecdf_on_grid(sample, grid):
    """ECDF evaluated on a grid: F_n(t) = (1/n) * #{x_i <= t}"""
    sample = np.sort(sample)
    return np.searchsorted(sample, grid, side="right") / sample.size


def make_grid(x, y, *, grid_mode="pooled", n_grid=400, max_grid=5000):
    """
    grid_mode:
      - "pooled": uses sorted unique pooled sample values (capped by max_grid)
      - "quantile": uses a quantile grid of pooled sample (size n_grid)
    """
    pooled = np.concatenate([x, y])
    pooled = pooled[np.isfinite(pooled)]
    pooled.sort()

    if grid_mode == "pooled":
        grid = np.unique(pooled)
        if grid.size > max_grid:
            idx = np.linspace(0, grid.size - 1, max_grid).astype(int)
            grid = grid[idx]
        return grid

    if grid_mode == "quantile":
        eps = 1e-6
        ps = np.linspace(eps, 1 - eps, int(n_grid))
        return np.quantile(pooled, ps)

    raise ValueError("grid_mode must be 'pooled' or 'quantile'.")


def dkw_band(F, n, *, alpha=0.05):
    """
    Dvoretzky-Kiefer-Wolfowitz (DKW) simultaneous band for a single ECDF:
        eps = sqrt(log(2/alpha)/(2n))
        band: F_n(x) +/- eps (clipped to [0, 1])
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    eps = np.sqrt(np.log(2.0 / alpha) / (2.0 * n))
    low = np.clip(F - eps, 0.0, 1.0)
    high = np.clip(F + eps, 0.0, 1.0)
    return low, high, eps


# ============================================================
# 1) QQ-plot + bootstrap bands (CI mode or Null (pooled) mode)
# ============================================================

def qqplot_with_bootstrap_bands(
    x,
    y,
    *,
    n_quantiles: int = 200,
    probs=None,
    n_boot: int = 1000,
    ci: float = 0.95,
    bootstrap_mode: str = "ci",   # "ci" or "null"
    random_state=None,
    ax=None,
    labels=("Sample A", "Sample B"),
    title="Empirical QQ-plot with bootstrap bands",
    add_45_line: bool = True,
    dropna: bool = True,
    band_alpha: float = 0.2,
):
    """
    QQ-plot between two empirical samples (1D, sizes can differ), with bootstrap bands.

    Two modes:
      - bootstrap_mode="ci": resample x and y separately. Bands approximate a pointwise CI
        for Delta(p)=Qy(p)-Qx(p) given the observed samples.
      - bootstrap_mode="null": pooled bootstrap under H0 (same distribution). Resample from
        pooled data, split into sizes nx and ny. Bands become a 'null envelope' centered near 0.

    Bands are built on Delta(p)=Qy(p)-Qx(p):
        band_low(p)  = Qx_obs(p) + Delta_low(p)
        band_high(p) = Qx_obs(p) + Delta_high(p)
    """
    x = _clean_1d(x, dropna=dropna)
    y = _clean_1d(y, dropna=dropna)

    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1).")
    if bootstrap_mode not in ("ci", "null"):
        raise ValueError("bootstrap_mode must be 'ci' or 'null'.")

    nx, ny = x.size, y.size

    # Probability grid
    if probs is None:
        eps = 0.5 / max(nx, ny)
        lo = max(eps, 1e-6)
        hi = 1.0 - lo
        probs = np.linspace(lo, hi, int(n_quantiles))
    else:
        probs = np.asarray(probs, dtype=float).ravel()
        if np.any(probs <= 0) or np.any(probs >= 1):
            raise ValueError("All probs must be strictly between 0 and 1.")

    # Observed quantiles
    qx = np.quantile(x, probs)
    qy = np.quantile(y, probs)

    # RNG
    gen = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    # Bootstrap deltas
    k = probs.size
    deltas = np.empty((int(n_boot), k), dtype=float)

    if bootstrap_mode == "ci":
        for b in range(int(n_boot)):
            xb = x[gen.integers(0, nx, size=nx)]
            yb = y[gen.integers(0, ny, size=ny)]
            deltas[b, :] = np.quantile(yb, probs) - np.quantile(xb, probs)

    else:  # "null"
        pooled = np.concatenate([x, y])
        n = pooled.size
        for b in range(int(n_boot)):
            boot = pooled[gen.integers(0, n, size=n)]  # pooled resample
            xb = boot[:nx]
            yb = boot[nx:nx+ny]
            deltas[b, :] = np.quantile(yb, probs) - np.quantile(xb, probs)

    alpha = 1.0 - ci
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    delta_low = np.percentile(deltas, lo_q, axis=0)
    delta_high = np.percentile(deltas, hi_q, axis=0)

    band_low = qx + delta_low
    band_high = qx + delta_high

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
    else:
        fig = ax.figure

    ax.fill_between(qx, band_low, band_high, alpha=band_alpha, linewidth=0)
    ax.scatter(qx, qy, s=18, alpha=0.85)

    if add_45_line:
        lo = np.nanmin([qx.min(), qy.min(), band_low.min()])
        hi = np.nanmax([qx.max(), qy.max(), band_high.max()])
        ax.plot([lo, hi], [lo, hi], linewidth=1.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel(f"Quantiles of {labels[0]}")
    ax.set_ylabel(f"Quantiles of {labels[1]}")
    ax.set_title(title + f"  (mode={bootstrap_mode})")
    ax.grid(True, alpha=0.3)

    result = {
        "probs": probs,
        "qx": qx,
        "qy": qy,
        "band_low": band_low,
        "band_high": band_high,
        "delta_ci": (delta_low, delta_high),
        "n_boot": int(n_boot),
        "ci": float(ci),
        "bootstrap_mode": bootstrap_mode,
    }
    return fig, ax, result


# ================================================================================
# 2) ECDF comparison + bands (bootstrap CI or Null pooled; DKW available as well)
# ================================================================================

def ecdfs_with_bands(
    x,
    y,
    *,
    grid_mode="pooled",
    n_grid=400,
    max_grid=5000,
    bands="bootstrap",            # "bootstrap", "dkw", or "both"
    n_boot=1000,
    ci=0.95,
    bootstrap_mode="ci",          # "ci" or "null"
    random_state=None,
    dropna=True,
    labels=("Sample A", "Sample B"),
    title="ECDF comparison with nonparametric bands",
    plot_diff=True,
    ax=None,
    band_alpha=0.20,
):
    """
    Plots two ECDFs on a common grid, optionally with bands.

    Bands:
      - "dkw": DKW simultaneous bands for each ECDF (no bootstrap).
      - "bootstrap": bootstrap bands (pointwise). Mode depends on bootstrap_mode:
           * bootstrap_mode="ci": resample x and y separately.
           * bootstrap_mode="null": pooled bootstrap under H0 (same distribution), resample pooled and split.
      - "both": draw both DKW and bootstrap bands.

    The difference panel plots Delta(x)=Fx-Fy with a band:
      - in "ci" mode: CI for the true Delta(x) given observed samples.
      - in "null" mode: a null envelope centered near 0 (what you'd expect if distributions are equal).
    """
    x = _clean_1d(x, dropna=dropna)
    y = _clean_1d(y, dropna=dropna)

    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1).")
    if bands not in ("bootstrap", "dkw", "both"):
        raise ValueError("bands must be 'bootstrap', 'dkw', or 'both'.")
    if bootstrap_mode not in ("ci", "null"):
        raise ValueError("bootstrap_mode must be 'ci' or 'null'.")

    want_boot = bands in ("bootstrap", "both")
    want_dkw = bands in ("dkw", "both")

    grid = make_grid(x, y, grid_mode=grid_mode, n_grid=n_grid, max_grid=max_grid)
    Fx = ecdf_on_grid(x, grid)
    Fy = ecdf_on_grid(y, grid)

    out = {
        "grid": grid,
        "Fx": Fx,
        "Fy": Fy,
        "ci": float(ci),
        "bands": bands,
        "bootstrap_mode": bootstrap_mode,
        "grid_mode": grid_mode,
    }

    # Axes
    if ax is None:
        if plot_diff:
            fig, axes = plt.subplots(
                2, 1, figsize=(7.5, 8.0), sharex=True,
                gridspec_kw={"height_ratios": [2.0, 1.2]}
            )
        else:
            fig, axes = plt.subplots(1, 1, figsize=(7.5, 5.5))
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]
        plot_diff = False

    ax0 = axes[0]

    alpha = 1.0 - ci

    # DKW bands
    if want_dkw:
        Fx_dkw_lo, Fx_dkw_hi, epsx = dkw_band(Fx, x.size, alpha=alpha)
        Fy_dkw_lo, Fy_dkw_hi, epsy = dkw_band(Fy, y.size, alpha=alpha)
        out["dkw"] = {
            "Fx_low": Fx_dkw_lo, "Fx_high": Fx_dkw_hi, "eps_x": epsx,
            "Fy_low": Fy_dkw_lo, "Fy_high": Fy_dkw_hi, "eps_y": epsy,
        }
        ax0.fill_between(grid, Fx_dkw_lo, Fx_dkw_hi, alpha=band_alpha, linewidth=0)
        ax0.fill_between(grid, Fy_dkw_lo, Fy_dkw_hi, alpha=band_alpha, linewidth=0)

    # Bootstrap bands
    if want_boot:
        gen = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        nx, ny = x.size, y.size
        k = grid.size

        Fx_boot = np.empty((int(n_boot), k), dtype=float)
        Fy_boot = np.empty((int(n_boot), k), dtype=float)

        if bootstrap_mode == "ci":
            for b in range(int(n_boot)):
                xb = x[gen.integers(0, nx, size=nx)]
                yb = y[gen.integers(0, ny, size=ny)]
                Fx_boot[b, :] = ecdf_on_grid(xb, grid)
                Fy_boot[b, :] = ecdf_on_grid(yb, grid)
        else:
            pooled = np.concatenate([x, y])
            n = pooled.size
            for b in range(int(n_boot)):
                boot = pooled[gen.integers(0, n, size=n)]
                xb = boot[:nx]
                yb = boot[nx:nx+ny]
                Fx_boot[b, :] = ecdf_on_grid(xb, grid)
                Fy_boot[b, :] = ecdf_on_grid(yb, grid)

        lo_q = 100.0 * (alpha / 2.0)
        hi_q = 100.0 * (1.0 - alpha / 2.0)

        Fx_lo = np.percentile(Fx_boot, lo_q, axis=0)
        Fx_hi = np.percentile(Fx_boot, hi_q, axis=0)
        Fy_lo = np.percentile(Fy_boot, lo_q, axis=0)
        Fy_hi = np.percentile(Fy_boot, hi_q, axis=0)

        out["bootstrap"] = {
            "Fx_low": Fx_lo, "Fx_high": Fx_hi,
            "Fy_low": Fy_lo, "Fy_high": Fy_hi,
            "n_boot": int(n_boot),
        }

        ax0.fill_between(grid, Fx_lo, Fx_hi, alpha=band_alpha, linewidth=0)
        ax0.fill_between(grid, Fy_lo, Fy_hi, alpha=band_alpha, linewidth=0)

        # Difference bands (CI or Null envelope, depending on bootstrap_mode)
        if plot_diff:
            Delta = Fx - Fy
            Delta_boot = Fx_boot - Fy_boot
            D_lo = np.percentile(Delta_boot, lo_q, axis=0)
            D_hi = np.percentile(Delta_boot, hi_q, axis=0)
            out["diff"] = {"Delta": Delta, "Delta_low": D_lo, "Delta_high": D_hi}

    # Plot ECDFs
    ax0.step(grid, Fx, where="post", linewidth=2.0, label=f"ECDF {labels[0]}")
    ax0.step(grid, Fy, where="post", linewidth=2.0, label=f"ECDF {labels[1]}")
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_title(title + f"  (mode={bootstrap_mode})")
    ax0.set_ylabel("F(x)")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # Difference panel
    if plot_diff:
        ax1 = axes[1]
        Delta = out.get("diff", {}).get("Delta", Fx - Fy)

        ax1.axhline(0.0, linewidth=1.5)
        if want_boot and "diff" in out:
            ax1.fill_between(grid, out["diff"]["Delta_low"], out["diff"]["Delta_high"],
                             alpha=band_alpha, linewidth=0)
        ax1.step(grid, Delta, where="post", linewidth=2.0)
        ax1.set_ylabel("Fx - Fy")
        ax1.set_xlabel("x")
        ax1.grid(True, alpha=0.3)

    return fig, axes, out


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    pass
    