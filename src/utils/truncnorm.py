from scipy.stats import truncnorm


def truncated_normal(
    size,
    threshold=2.0,
):
    x = truncnorm.rvs(-threshold, threshold, size=size)
    return x
