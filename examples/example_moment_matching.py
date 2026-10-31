import numpy as np
from scipy import stats

from snpdenspy.snp.snp_distribution import SNPDistribution
from snpdenspy.util.diagnose.diagnose_fit import diagnose_fit

nig_dist = stats.norminvgauss(a=3.0, b=1.0, loc=5.0, scale=3.0)
x = nig_dist.rvs(size=4000, random_state=np.random.default_rng(98))
        
snp = SNPDistribution(K=6)
c_hat, loc_hat, scale_hat = snp.fit_moment_matching(x=x, fit_loc_scale=True)        
fitted_dist = snp.freeze(c=c_hat, loc=loc_hat, scale=scale_hat)

print(f"\n=== NIG(a=3, b=1) ===")
diagnose_fit(
    x=x,
    fitted_dist=fitted_dist,
    pdf_original_func=nig_dist.pdf,
    title="NIG(a=3, b=1)",
    compare_normal=False,
    compare_kde=True)

print("\n=== Done ===")
