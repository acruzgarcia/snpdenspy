
from snpdenspy.snp.snp_distribution import SNPDistribution
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ar1_loc_scale(theta, x):
    phi, log_sigma = theta
    y = np.asarray(x)
    loc = phi * np.concatenate([[0.0], y[:-1]])  # simple AR(1) on lagged y
    scale = np.exp(log_sigma)
    return loc, scale

def ar1_gaussian_mle(x):
    """
    MLE for AR(1) with Gaussian noise and zero intercept.
    """
    y = np.asarray(x, dtype=float)
    x_lag = y[:-1]
    x_curr = y[1:]
    denom = float(np.dot(x_lag, x_lag))
    if denom == 0.0:
        raise ValueError("Cannot estimate AR(1) when lagged values are all zero.")
    phi_hat = float(np.dot(x_curr, x_lag) / denom)
    resid = x_curr - phi_hat * x_lag
    sigma_hat = float(np.sqrt(np.mean(resid ** 2)))
    return phi_hat, sigma_hat


if __name__ == "__main__":
    
    from scipy.stats import t
    from snpdenspy.util.diagnose.diagnose_fit import diagnose_fit

    # Parameters
    phi = 0.7
    sigma = 0.2
    nsamples = 10000

    # Generate noise
    np.random.seed(42)
    noise = t.rvs(df=7, size=nsamples)
    noise = (noise - noise.mean()) / noise.std(ddof=1)  # standardize to mean 0 and variance 1
    
    # First simulate the distribution according to the parameters
    x = np.zeros(shape=(noise.shape))
    x[0] = noise[0]
    for i in range(1, len(x)):
        x[i] = phi*x[i-1] + sigma * noise[i]

    phi_norm, sigma_norm = ar1_gaussian_mle(x)
    print(f"Normal-noise AR(1) MLE: phi={phi_norm:.4f}, sigma={sigma_norm:.4f}")

    # Now, we fit
    K = 6
    dist = SNPDistribution(K=K)
    theta0 = np.array([0.1, np.log(np.std(x, ddof=1))])

    res = dist.fit_mle(
        x=x,
        fit_loc=False, fit_scale=False,          # loc/scale driven by model
        loc_scale_fn=ar1_loc_scale,
        constrain_std_moments=True,
        loc_scale_params0=theta0,
        return_opt_result=True,
        method="L-BFGS-B",
    )
    c_hat = res["c"]
    theta_hat = res["external_params"]
    phi_snp, sigma_snp = theta_hat[0], np.exp(theta_hat[1])
    loc_hat, scale_hat = res["loc"], res["scale"]
    print(f"SNP-noise AR(1) MLE: phi={phi_snp:.4f}, sigma={sigma_snp:.4f}")

    diagnose_fit(x=noise, fitted_dist=SNPDistribution(K=K)(c=c_hat))

    # QQ plots for residuals vs assumed Normal and fitted SNP noise
    eps_norm = (x[1:] - phi_norm * x[:-1]) / sigma_norm
    eps_snp = (x[1:] - phi_snp * x[:-1]) / sigma_snp

    probs = (np.arange(1, len(eps_norm) + 1) - 0.5) / len(eps_norm)
    emp_norm = np.sort(eps_norm)
    emp_snp = np.sort(eps_snp)

    q_norm = stats.norm.ppf(probs)
    q_snp = SNPDistribution(K=K)(c=c_hat).ppf(probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(q_norm, emp_norm, "o", markersize=3, alpha=0.6, label="Normal fit")
    ax.plot(q_snp, emp_snp, "x", markersize=3, alpha=0.6, label="SNP fit")
    q_min = min(q_norm.min(), emp_norm.min(), q_snp.min(), emp_snp.min())
    q_max = max(q_norm.max(), emp_norm.max(), q_snp.max(), emp_snp.max())
    ax.plot([q_min, q_max], [q_min, q_max], "--", color="gray", linewidth=1)
    ax.set_title("QQ: standardized residuals")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical residual quantiles")
    ax.legend()

    fig.tight_layout()
    plt.show()

    
