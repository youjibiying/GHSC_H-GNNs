from scipy import stats
import numpy as np

def is_significantly_greater(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
    """
    check the second one if significantly larger than the first one.

    return:
    - significant: 
    - p_value: p 
    """

    t_stat, p_value = stats.ttest_ind_from_stats(
        mean1=mean1, std1=std1, nobs1=n1,
        mean2=mean2, std2=std2, nobs2=n2,
        equal_var=False, alternative='less'
    )

    significant = p_value / 2 < alpha and t_stat < 0

    return significant, p_value / 2

# 示例用法
mean1 = 91.99
std1 = 0.16

n1 = 10

mean2 = 92.04
std2 = 0.07
n2 = 10

significant, p_value = is_significantly_greater(mean1, std1, n1, mean2, std2, n2)
print(f"Significant: {significant}, P-value: {p_value:.4f}")