# from .distributions import *
# from . import nbinom
from .vm import *
from .digest import *
from .nodes import *
from .opcodes import *
import numpy as np

import builtins

# __version__ = '0.2.0.post7.dev0+3e6fe06'

# def max(val1, val2):
#     """
#     Returns the distribution of the max of one or more distributions.
#     """
#     return val1.__max__(val2)
    # val1 = args[0]

    # for val2 in args[1:]:
    #     val1 = val1.__max__(val2)

    # return val1



def P(rv):
    """
    Returns the probability that a random var is greater than 0.
    """
    if hasattr(rv, 'cdf'):
        return 1 - rv.cdf(0.5)
    else:
        rv_ = rv.compute()
        # print(rv_._digests.getBins())
        # print(rv_._digests.getWeights())        
        return 1 - rv_.cdf(0.5)


def E(rv):
    """
    Returns the expected value of the random variable.
    """
    return rv.mean()

    

def max(*args):
    """Return the maximum of multiple random variables or expressions.
    
    Computes the element-wise maximum across multiple random variables by
    sequentially applying the __max__ method. This creates an expression tree
    that will evaluate to the maximum value during Monte Carlo simulation.
    
    Args:
        *args: Variable number of RandomVariable instances or numeric values
        
    Returns:
        RandomVariable: Expression representing the maximum of all inputs
        
    Example:
        >>> x = Normal(0, 1)
        >>> y = Normal(2, 1)
        >>> z = Normal(-1, 1)
        >>> max_expr = max(x, y, z)  # Maximum of three normal distributions
        >>> result = max_expr.compute()  # Monte Carlo simulation
    """
    val1 = args[0]

    for val2 in args[1:]:
        val1 = Max(val1, val2)
        #val1 = val1.__max__(val2)

    return val1


def min(*args):
    """Return the minimum of multiple random variables or expressions.
    
    Computes the element-wise minimum across multiple random variables by
    sequentially applying the __min__ method. This creates an expression tree
    that will evaluate to the minimum value during Monte Carlo simulation.
    
    Args:
        *args: Variable number of RandomVariable instances or numeric values
        
    Returns:
        RandomVariable: Expression representing the minimum of all inputs
        
    Example:
        >>> x = Normal(0, 1)
        >>> y = Normal(2, 1)
        >>> z = Normal(-1, 1)
        >>> min_expr = min(x, y, z)  # Minimum of three normal distributions
        >>> result = min_expr.compute()  # Monte Carlo simulation
    """
    val1 = args[0]

    for val2 in args[1:]:
        val1 = Min(val1, val2)
        #val1 = val1.__min__(val2)

    return val1


def printPMF(rv):

    kVec = []
    pVec = []

    cumProb = 0
    k = rv.lowerBound()
    while cumProb < UPPER:
        p = rv.pmf(k)
        kVec.append(k)
        pVec.append(p)
        cumProb += rv.pmf(k)
        k += 1

    maxP = builtins.max(pVec)

    for k, p in zip(kVec, pVec):
        print(f'{k:5} | {"█"*int(p / maxP *100)}')

def fromScipy(rvScipy, maxBins=32, samples=10_000):
    """
    Generates a RandomVariable by repeatably sampling a Scipy frozen distribution
    this allows us to use Scipy and Statsmodels to fit distributions and then
    convert them to Casino Random Variables.
    """
    data = rvScipy.rvs(size=samples)
    return fromArray(data)
    counts = np.ones_like(data)
    rv = RandomVariable(maxBins=maxBins, data=data, counts=counts)
    return rv


def fromArray(array, maxBins=32):
    """Create a DigestVariable from an array of data points.
    
    Constructs a t-digest from empirical data and wraps it in a DigestVariable
    for use in algebraic operations and further Monte Carlo modeling. This allows
    historical data or simulation results to be incorporated into new probabilistic
    models.
    
    The t-digest provides an efficient approximation of the data distribution
    with bounded memory usage, making it suitable for large datasets while
    preserving accurate quantile estimates.
    
    Args:
        array (array-like): Numerical data points to fit the digest to
        maxBins (int, optional): Maximum number of centroids in the t-digest.
                               Defaults to 32. Higher values provide better
                               accuracy at the cost of memory usage.
    
    Returns:
        DigestVariable: A random variable representing the empirical distribution
                       that can be used in further algebraic operations
    
    Example:
        >>> import numpy as np
        >>> historical_returns = np.random.normal(0.05, 0.2, 1000)
        >>> return_dist = fromArray(historical_returns)
        >>> future_value = 1000 * (1 + return_dist)  # Model future portfolio value
        >>> result = future_value.compute()
    """
    digest = Digest(maxBins=maxBins)
    digest.fit(array)
    # for ar in array:
    #     digest.add(ar)

    rv = DigestVariable(digest)
    return rv



def plot(digest, nBins=20, lower=None, upper=None, width=0.8, ax=None, *args, **kwds):
    """Plot a histogram representation of a digest distribution.
    
    Creates a bar chart visualization of the probability distribution represented
    by a Digest or DigestVariable. The plot shows the probability density across
    bins by computing the difference in cumulative distribution function (CDF)
    values between bin boundaries.
    
    This function is useful for visualizing the results of Monte Carlo simulations
    and understanding the shape of distributions created through algebraic
    operations on random variables.
    
    Args:
        digest (Digest or DigestVariable): The digest object to plot
        nBins (int, optional): Number of histogram bins to create. Defaults to 20.
        lower (float, optional): Lower bound for the plot range. If None, uses
                                digest.lower(). Defaults to None.
        upper (float, optional): Upper bound for the plot range. If None, uses
                                digest.upper(). Defaults to None.
        width (float, optional): Width factor for bars relative to bin width.
                               Defaults to 0.8.
        ax (matplotlib.Axes, optional): Matplotlib axes to plot on. If None,
                                      creates a new figure and axes. Defaults to None.
        *args: Additional positional arguments passed to matplotlib's bar() function
        **kwds: Additional keyword arguments passed to matplotlib's bar() function
    
    Returns:
        None: The function modifies the provided axes or creates a new plot
    
    Example:
        >>> x = Normal(0, 1)
        >>> result = x.compute(samples=10000)
        >>> plot(result, nBins=30)  # Plot with 30 bins
        >>> 
        >>> # Custom styling
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> plot(result, nBins=25, ax=ax, color='red', alpha=0.7)
        >>> ax.set_title('Distribution Results')
    """
    if lower is None:
        lower = digest.lower()

    if upper is None:
        upper = digest.upper()

    bins = np.linspace(lower, upper, nBins)
    y = np.zeros(nBins - 1)
    x = np.zeros(nBins - 1)    

    for i in range(nBins - 1):
        y[i] = digest.cdf(bins[i+1]) - digest.cdf(bins[i])
        x[i] = (bins[i+1] + bins[i])/2

    if ax is None:
        import pylab as plt
        fig = plt.figure()
        ax = fig.gca()

    ax.bar(x, y, width=width*(bins[1] - bins[0]), *args, **kwds)
        

    
    
