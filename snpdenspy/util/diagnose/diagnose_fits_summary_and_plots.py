
import numpy as np
import matplotlib.pyplot as plt
    
from snpdenspy.snp.snp_distribution import SNPDistribution
from snpdenspy.util.statistical_testing.statistics import (
    ks_statistic_1d_sample_model,
    cvm_statistic_1d_sample_model
)


def _print_parameter_summary(fit_dist_mm: SNPDistribution, fit_dist_mle: SNPDistribution):
    
    print("\n--- Moment-matching and MLE fit summary ---")
    print("\nFitted parameters:")
    print(
        "  Moment-matching:",
        f"c={np.array2string(fit_dist_mm.c, precision=4, floatmode='fixed')}",
        f"loc={fit_dist_mm.loc:.4f}",
        f"scale={fit_dist_mm.scale:.4f}",
    )
    print(
        "  MLE:            ",
        f"c={np.array2string(fit_dist_mle.c, precision=4, floatmode='fixed')}",
        f"loc={fit_dist_mle.loc:.4f}",
        f"scale={fit_dist_mle.scale:.4f}",
    )


def _print_gof_summary(x, fit_dist_mm: SNPDistribution, fit_dist_mle: SNPDistribution):

    ll_mm = fit_dist_mm.loglik(x=x)
    ll_mle = fit_dist_mle.loglik(x=x)

    ks_mm = ks_statistic_1d_sample_model(x, fit_dist_mm.cdf)
    cvm_mm = cvm_statistic_1d_sample_model(x, fit_dist_mm.cdf)

    ks_mle = ks_statistic_1d_sample_model(x, fit_dist_mle.cdf)
    cvm_mle = cvm_statistic_1d_sample_model(x, fit_dist_mle.cdf)

    print(f"\nLog-likelihood (MM fit):  {ll_mm:12.4f}")
    print(f"Log-likelihood (MLE fit): {ll_mle:12.4f}")
    print("\nTwo-sample statistics vs fitted draws (empirical vs fitted):")
    print(f"  Moment-matching: KS={ks_mm:8.4f}  CVM={cvm_mm:8.4f}")
    print(f"  MLE:             KS={ks_mle:8.4f}  CVM={cvm_mle:8.4f}")


def _print_moment_table(x, fit_dist_mm: SNPDistribution, fit_dist_mle: SNPDistribution, K):

    header = f"{'Moment order':>2}  {'empirical raw moment':>18}  {'MM fitted raw moment':>12} {'MLE fitted raw moment':>12}"
    lines = [header]
    lines.append("-" * (len(header) + 5))
    for order in range(1, K + 3):
        lines.append(
            f"{order:2d}  {np.mean(x ** order):18.4f}  {fit_dist_mm.moment(order=order):12.4f} {fit_dist_mle.moment(order=order):12.4f}"
        )
    print("\n".join(lines))


def _print_quantile_table(x, fit_dist_mm: SNPDistribution, fit_dist_mle: SNPDistribution):
    
    probs = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    emp_q = np.quantile(x, probs)
    mm_q = fit_dist_mm.ppf(probs)
    mle_q = fit_dist_mle.ppf(probs)

    header = f"{'q':>5} {'Empirical':>12} {'MM fitted':>12} {'MLE fitted':>12}"
    lines = [header, "-" * (len(header) + 5)]
    for p, e, m1, m2 in zip(probs, emp_q, mm_q, mle_q):
        lines.append(f"{int(p*100):3d}% {e:12.4f} {m1:12.4f} {m2:12.4f}")
    print("\nQuantile comparison (empirical vs fitted):")
    print("\n".join(lines))
    

def _plot_density_and_qq(x, xs, pdf_original, pdf_fitted_mm, pdf_fitted_mle, fit_dist_mm, fit_dist_mle, title):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # density plot
    ax0 = axes[0]
    ax0.hist(x, bins=50, density=True, alpha=0.6, color="orange", label="PDF")
    ax0.plot(xs, pdf_fitted_mm, label="Moment-matching", color="blue", lw=2)
    ax0.plot(xs, pdf_fitted_mle, label="MLE", color="black", lw=2)
    ax0.plot(xs, pdf_original, label="Original", color="red", lw=2)
    ax0.set_title(title or "Empirical, Moment-matching and MLE fit")
    ax0.set_xlabel("x")
    ax0.set_ylabel("Density")
    ax0.legend()

    # qq plot: theoretical on x-axis, empirical on y-axis using full sample
    n = len(x)
    probs = (np.arange(1, n + 1) - 0.5) / n
    emp_sorted = np.sort(x)
    mm_q = fit_dist_mm.ppf(probs)
    mle_q = fit_dist_mle.ppf(probs)

    ax1 = axes[1]
    ax1.plot(mm_q, emp_sorted, "o", label="Moment-matching", alpha=0.5, color="blue", markersize=3)
    ax1.plot(mle_q, emp_sorted, "x", label="MLE", alpha=0.5, color="black", markersize=3)
    # 45-degree reference line
    all_theor = np.concatenate([mm_q, mle_q])
    q_min = min(all_theor.min(), emp_sorted.min())
    q_max = max(all_theor.max(), emp_sorted.max())
    ax1.plot([q_min, q_max], [q_min, q_max], "--", color="gray", linewidth=1)
    ax1.set_title("QQ plot (sample distribution vs fitted ones)")
    ax1.set_xlabel("Sample quantiles")
    ax1.set_ylabel("Fitted quantiles")
    ax1.legend()

    fig.tight_layout()
    plt.show()

def diagnose_snp_fits(
        sample_func: callable,
        pdf_func: callable,
        K: int,
        fit_loc_scale: bool,
        nsamples: int,
        rng = None,
        title: str | None = None,
    ):

    if rng is None:
        rng = np.random.default_rng()

    x = sample_func(nsamples=nsamples, rng=rng)

    snp = SNPDistribution(K=K)

    # Fit empirical distribution by moment-matching and MLE
    c_fit_mm, loc_fit_mm, scale_fit_mm = snp.fit_moment_matching(
        x=x, fit_loc_scale=fit_loc_scale)

    c_fit_mle, loc_fit_mle, scale_fit_mle = snp.fit_mle(
        x=x, fit_loc=fit_loc_scale, fit_scale=fit_loc_scale)
    
    fit_dist_mm = snp(c=c_fit_mm, loc=loc_fit_mm, scale=scale_fit_mm)
    fit_dist_mle = snp(c=c_fit_mle, loc=loc_fit_mle, scale=scale_fit_mle)

    # textual diagnostics
    _print_parameter_summary(fit_dist_mm, fit_dist_mle)
    _print_gof_summary(x, fit_dist_mm, fit_dist_mle)
    _print_moment_table(x, fit_dist_mm, fit_dist_mle, K)
    _print_quantile_table(x, fit_dist_mm, fit_dist_mle)

    #  Grid ---
    margin =  0.05 * (np.max(x) - np.min(x))
    xs = np.linspace(np.min(x) - margin , np.max(x) + margin, 500)

    pdf_original = pdf_func(xs)
    pdf_fitted_mm = fit_dist_mm.pdf(xs)
    pdf_fitted_mle = fit_dist_mle.pdf(xs)

    _plot_density_and_qq(
        x=x,
        xs=xs,
        pdf_original=pdf_original,
        pdf_fitted_mm=pdf_fitted_mm,
        pdf_fitted_mle=pdf_fitted_mle,
        fit_dist_mm=fit_dist_mm,
        fit_dist_mle=fit_dist_mle,
        title=title,
    )
