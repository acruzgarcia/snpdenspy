
import numpy as np
import matplotlib.pyplot as plt
    
from snpdenspy.snp.snp_distribution import SNPDistribution
from snpdenspy.util.statistical_testing.statistics import (
    ks_statistic_1d_sample_model,
    cvm_statistic_1d_sample_model
)


def _print_parameter_summary(fit_dist: SNPDistribution):
    
    print("Fitted parameters:")
    fmt3 = lambda x: np.format_float_positional(  # noqa: E731
        x, precision=3, unique=False, fractional=True, trim="k"
    )
    fmt4 = lambda x: np.format_float_positional(  # noqa: E731
        x, precision=4, unique=False, fractional=True, trim="k"
    )
    print(
        f"c={np.array2string(fit_dist.c, formatter={'float_kind': fmt3})}",
        f"loc={fmt4(fit_dist.loc)}",
        f"scale={fmt4(fit_dist.scale)}",
    )

def _print_gof_summary(x, fit_dist: SNPDistribution):

    ll = fit_dist.loglik(x=x)
    
    ks = ks_statistic_1d_sample_model(x, fit_dist.cdf)
    cvm = cvm_statistic_1d_sample_model(x, fit_dist.cdf)

    print(f"Log-likelihood:  {ll:12.4f}")
    print("Two-sample statistics vs fitted draws (empirical vs fitted):")
    print(f"  KS={ks:8.4f}  CVM={cvm:8.4f}")

def _print_moment_table(x, fit_dist: SNPDistribution, K):

    header = f"{'Moment order':>2}  {'Empirical raw moment':>18}  {'Fitted raw moment':>12}"
    lines = [header]
    lines.append("-" * (len(header) + 5))
    for order in range(1, K + 3):
        lines.append(
            f"{order:2d}  {np.mean(x ** order):18.4f}  {fit_dist.moment(order=order):12.4f}"
        )
    print("\n".join(lines))


def _print_quantile_table(x, fit_dist: SNPDistribution):
    
    probs = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    emp_q = np.quantile(x, probs)
    fit_q = fit_dist.ppf(probs)
    
    header = f"{'q':>5} {'Empirical':>12} {'Fitted':>12}"
    lines = [header, "-" * (len(header) + 5)]
    for p, e, m in zip(probs, emp_q, fit_q):
        lines.append(f"{int(p*100):3d}% {e:12.4f} {m:12.4f}")
    print("\n".join(lines))
    

def _plot_density_and_qq(x, xs, pdf_original, pdf_fitted, fit_dist, title):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # density plot
    ax0 = axes[0]
    ax0.hist(x, bins=50, density=True, alpha=0.6, color="orange", label="PDF")
    ax0.plot(xs, pdf_fitted, label="Fitted", color="blue", lw=2)
    ax0.plot(xs, pdf_original, label="Original", color="red", lw=2)
    ax0.set_title(title or "Empirical and Fitted")
    ax0.set_xlabel("x")
    ax0.set_ylabel("Density")
    ax0.legend()

    # qq plot: theoretical on x-axis, empirical on y-axis using full sample
    n = len(x)
    probs = (np.arange(1, n + 1) - 0.5) / n
    emp_sorted = np.sort(x)
    theor_q = fit_dist.ppf(probs)
    
    ax1 = axes[1]
    ax1.plot(theor_q, emp_sorted, "o", label="Fit", alpha=0.5, color="blue", markersize=3)
    # 45-degree reference line
    q_min = min(theor_q.min(), emp_sorted.min())
    q_max = max(theor_q.max(), emp_sorted.max())
    ax1.plot([q_min, q_max], [q_min, q_max], "--", color="gray", linewidth=1)
    ax1.set_title("QQ plot (sample distribution vs fitted one)")
    ax1.set_xlabel("Sample quantiles")
    ax1.set_ylabel("Fitted quantiles")
    ax1.legend()

    fig.tight_layout()
    plt.show()

def diagnose_fit(
        sample_func: callable,
        pdf_original_func: callable,
        K: int,
        fit_loc_scale: bool,
        nsamples: int,
        method: str = 'mle', #{'mle', 'mm'} 
        rng = None,
        title: str | None = None,
    ):

    if rng is None:
        rng = np.random.default_rng()

    x = sample_func(nsamples=nsamples, rng=rng)

    snp = SNPDistribution(K=K)

    # Fit empirical distribution by moment-matching or MLE
    
    c_fit, loc_fit, scale_fit = snp.fit_moment_matching(
        x=x, fit_loc_scale=fit_loc_scale) if method == 'mm' else snp.fit_mle(
            x=x, fit_loc=fit_loc_scale, fit_scale=fit_loc_scale)
        
    fit_dist = snp(c=c_fit, loc=loc_fit, scale=scale_fit)
    
    # textual diagnostics
    print("\n--- Fit summary ---")
    _print_parameter_summary(fit_dist)
    _print_gof_summary(x, fit_dist)
    _print_moment_table(x, fit_dist, K)
    _print_quantile_table(x, fit_dist)

    #  Grid ---
    margin =  0.05 * (np.max(x) - np.min(x))
    xs = np.linspace(np.min(x) - margin , np.max(x) + margin, 500)

    pdf_original = pdf_original_func(xs)
    pdf_fitted = fit_dist.pdf(xs)
    
    _plot_density_and_qq(
        x=x,
        xs=xs,
        pdf_original=pdf_original,
        pdf_fitted=pdf_fitted,
        fit_dist=fit_dist,
        title=title,
    )
