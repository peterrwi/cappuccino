import numpy as np

def mplFontScales(fontsize):
    font_scalings = {
        'xx-small' : 0.579,
        'x-small'  : 0.694,
        'small'    : 0.833,
        'medium'   : 1.0,
        'large'    : 1.200,
        'x-large'  : 1.440,
        'xx-large' : 1.728,
        'larger'   : 1.2,
        'smaller'  : 0.833,
        None       : 1.0}
    return 10. * font_scalings[fontsize]

def errorsum(x):
    """
    Helper function for adding uncertainties

    :param x: List of uncertainties
    :type x: list of floats

    :return: error
    :rtype: float
    """
    error_sum = np.sqrt(sum(x[i] ** 2 for i in range(len(x))))
    return error_sum


def findDimensions(n):
    """
    Automatically determine number of subplot columns and rows based on the total
    number of panels. Aims for either ncol = nrow or ncol = nrow + 1.

    :param n: Total number of panels
    :type n: int

    :return: error
    :rtype: float
    """
    for i in range(10):
        if n <= i * i:
            return (i, i)
        if n <= i * (i + 1):
            return (i, i + 1)


def confInt(x, percentile=68.27, onesided=False, prior=None):
    """
    Calculates the confidence interval for a posterior sample x.

    :param x: 1D array of the posterior samples
    :type x: numpy array
    
    :param percentile: Percentile for which to calculate confidence
        interval. Defaults to 68.27.
    :type percentile: float

    :param onesided: Standard confidence interval (False, None), upper limit
        ('lower', 'hi'), or lower limit ('upper', 'lo')
    :type onesided: float

    :param prior: Prior range for the parameter, recommended if 'onesided'
        is not False.
    :type prior: list, length 2

    :return: Confidence interval in a numpy array, formatted as [median, 
      lower limit, upper limit]
    """
    if onesided in [False, None]:
        med, hi, lo = np.percentile(
            x,
            (50, 50+percentile/2., 50-percentile/2.)
        )
        confint = np.array([med, med-lo, hi-med])
    elif onesided in ['lo','upper']:
        if prior != None:
            lower = prior[0]
        else:
            lower = min(x)
        val = np.percentile(x, percentile)
        confint = np.array([lower, 0, val - lower])
    elif onesided in ['hi','lower']:
        if prior != None:
            upper = prior[1]
        else:
            upper = max(x)
        val = np.percentile(x, 100.0 - percentile)
        confint = np.array([upper, upper - val, 0])    

    return confint