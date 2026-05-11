from typing import SupportsAbs
import numpy as np
from .digest_cpp import Digest
from .vm import VirtualMachine
from .opcodes import *

class RandomVariable():
    """
    Base class for building expression trees representing random variables.
    
    This class serves as the foundation for a domain-specific language (DSL) that allows
    users to define complex probabilistic models using familiar mathematical operators.
    Random variables can be combined using arithmetic operations, comparisons, and
    summations to create sophisticated Monte Carlo simulation models.
    
    The class implements operator overloading to enable natural mathematical syntax:
    - Arithmetic: +, -, *, /, //, %, **
    - Comparisons: <, <=, >, >=
    - Matrix multiplication (@) for summation operations
    - Negation and absolute value
    
    When computed, each RandomVariable generates a Monte Carlo simulation that
    produces a Digest (t-digest) summarizing the output distribution.
    
    Example:
        >>> x = Normal(0, 1)
        >>> y = Normal(2, 1)  
        >>> z = x + y  # Creates expression tree
        >>> result = z.compute()  # Runs Monte Carlo simulation
        >>> result.quantile(0.5)  # Get median
    """
    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __neg__(self):
        return Neg(self)

    def __abs__(self):
        return Abs(self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __floordiv__(self, other):
        return FloorDiv(self, other)

    def __rfloordiv__(self, other):
        return FloorDiv(other, self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    def __matmul__(self, other):
        return Sum(self, other)

    def __lt__(self, other):
        return Lt(self, other)

    def __le__(self, other):
        return Lte(self, other)

    def __gt__(self, other):
        return Gt(self, other)

    def __ge__(self, other):
        return Gte(self, other)

    def __max__(self, other):
        return Max(self, other)

    def __min__(self, other):
        return Min(self, other)


class Constant(RandomVariable):
    """
    A random variable that always returns a fixed constant value.
    
    This class wraps a constant numeric value in the RandomVariable interface,
    allowing constants to participate in algebraic expressions with other
    random variables.
    
    Args:
        value: The constant numeric value
        
    Example:
        >>> c = Constant(5)
        >>> x = Normal(0, 1)
        >>> result = (x + c).compute()  # Shifts normal by 5
    """
    def __init__(self, value):
        self.value = value

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm.codes    = np.append(vm.codes,    _PUSH)
        vm.operands = np.append(vm.operands, self.value)
        return vm


class BinaryOp(RandomVariable):
    """
    Base class for binary operations on random variables.
    
    Represents an operation applied to two operands (left and right), which
    can be either RandomVariable instances or constants. Constants are
    automatically wrapped in Constant nodes.
    
    Args:
        left: Left operand (RandomVariable or numeric)
        right: Right operand (RandomVariable or numeric)
    """
    def __init__(self, left, right):
        if not isinstance(left, RandomVariable):
            left  = Constant(left)
        if not isinstance(right, RandomVariable):
            right = Constant(right)

        self.left  = left
        self.right = right

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm = self.left._compile(vm)
        vm = self.right._compile(vm)
        vm.codes    = np.append(vm.codes,    self.opCode)
        vm.operands = np.append(vm.operands, 0)
        return vm


class Add(BinaryOp):
    opCode = _ADD

class Sub(BinaryOp):
    opCode = _SUB

class Mul(BinaryOp):
    opCode = _MUL

class Div(BinaryOp):
    opCode = _DIV

class Mod(BinaryOp):
    opCode = _MOD

class FloorDiv(BinaryOp):
    opCode = _FLOORDIV

class Pow(BinaryOp):
    opCode = _POW

class Max(BinaryOp):
    opCode = _MAX

class Min(BinaryOp):
    opCode = _MIN

class Gt(BinaryOp):
    opCode = _GT

class Gte(BinaryOp):
    opCode = _GTE

class Lt(BinaryOp):
    opCode = _LT

class Lte(BinaryOp):
    opCode = _LTE


class UnaryOp(RandomVariable):
    """
    Base class for unary operations on random variables.
    
    Represents an operation applied to a single operand, which
    can be either a RandomVariable instance or a constant.
    Constants are automatically wrapped in Constant nodes.
    
    Args:
        operand: The operand (RandomVariable or numeric)
    """
    def __init__(self, operand):
        if not isinstance(operand, RandomVariable):
            operand = Constant(operand)
        self.operand = operand

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm = self.operand._compile(vm)
        vm.codes    = np.append(vm.codes,    self.opCode)
        vm.operands = np.append(vm.operands, 0)
        return vm


class Neg(UnaryOp):
    opCode = _NEG

class Abs(UnaryOp):
    opCode = _ABS


class Sum(RandomVariable):
    """Sum of multiple independent samples from a distribution.
    
    The Sum node implements the @ operator (matrix multiplication) to create
    a new random variable representing the sum of n independent samples from
    the distribution. This is useful for modeling aggregate quantities like
    total claims, portfolio returns, or cumulative effects.
    
    Args:
        left (RandomVariable): The distribution to sample from
        right (int or RandomVariable): The number of samples to sum
        
    Example:
        >>> d = RandInt(1, 6)  # Die roll
        >>> total = d @ 5      # Sum of 5 dice
        >>> result = total.compute()
    """
    def __init__(self, left, right):
        self.left  = left
        self.right = right

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))

        n = self.right

        for _ in range(n):
            vm = self.left._compile(vm)
            if _ > 0:
                vm.codes    = np.append(vm.codes,    _ADD)
                vm.operands = np.append(vm.operands, 0)

        return vm


class RandInt(RandomVariable):
    """Random integer uniformly distributed between low and high (inclusive).
    
    Generates random integers from a discrete uniform distribution over
    the range [low, high]. This is useful for modeling dice rolls, random
    counts, or any discrete uniform random variable.
    
    Args:
        low (int): Minimum value (inclusive)
        high (int): Maximum value (inclusive)
        
    Example:
        >>> die = RandInt(1, 6)  # Standard six-sided die
        >>> result = die.compute(samples=10000)
        >>> result.quantile(0.5)  # Approximately 3 or 4
    """
    def __init__(self, low, high):
        self.low  = low
        self.high = high

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm.codes    = np.append(vm.codes,    [_PUSH, _PUSH, _RANDINT])
        vm.operands = np.append(vm.operands, [self.low, self.high, 0])
        return vm


class Normal(RandomVariable):
    """Random variable following a normal (Gaussian) distribution.
    
    Generates samples from a normal distribution with specified mean and
    standard deviation. Useful for modeling quantities that cluster around
    a central value with symmetric variation.
    
    Args:
        mu (float): Mean (location parameter)
        sigma (float): Standard deviation (scale parameter, must be > 0)
        
    Example:
        >>> height = Normal(170, 10)  # Height in cm
        >>> result = height.compute()
        >>> result.quantile(0.5)  # Approximately 170
    """
    def __init__(self, mu, sigma):
        self.mu    = mu
        self.sigma = sigma

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm.codes    = np.append(vm.codes,    [_PUSH, _PUSH, _RANDNORM])
        vm.operands = np.append(vm.operands, [self.mu, self.sigma, 0])
        return vm


class Exp(RandomVariable):
    """Random variable following an exponential distribution.
    
    Generates samples from an exponential distribution with specified rate
    parameter (lambda). Useful for modeling waiting times, service times,
    or decay processes.
    
    Args:
        lam (float): Rate parameter (lambda, must be > 0). The mean of
                    the distribution is 1/lam.
        
    Example:
        >>> wait_time = Exp(0.5)  # Average wait of 2 units
        >>> result = wait_time.compute()
    """
    def __init__(self, lam):
        self.lam = lam

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm.codes    = np.append(vm.codes,    [_PUSH, _RANDEXP])
        vm.operands = np.append(vm.operands, [self.lam, 0])
        return vm


class Gamma(RandomVariable):
    """Random variable following a gamma distribution.
    
    Generates samples from a gamma distribution with shape k, scale theta,
    and optional location parameter.
    
    Args:
        k (float): Shape parameter (must be > 0)
        theta (float): Scale parameter (must be > 0). Mean = k * theta.
        loc (float, optional): Location (shift) parameter. Defaults to 0.
        
    Example:
        >>> x = Gamma(2.0, 3.0)  # Shape=2, Scale=3, mean=6
        >>> result = x.compute()
        >>> result.mean()  # Approximately 6
    """
    def __init__(self, k, theta, loc=0):
        self.k     = k
        self.theta = theta
        self.loc   = loc

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm.codes    = np.append(vm.codes,    [_PUSH, _PUSH, _PUSH, _RANDGAMMA])
        vm.operands = np.append(vm.operands, [self.k, self.theta, self.loc, 0])
        return vm


class Poisson(RandomVariable):
    """Random variable following a Poisson distribution.
    
    Generates samples from a Poisson distribution with specified rate
    parameter lambda. Useful for modeling count data or rare events.
    
    Args:
        lam (float): Rate parameter (lambda, must be > 0). Both the mean
                    and variance of the distribution equal lambda.
        
    Example:
        >>> arrivals = Poisson(3.5)  # Average 3.5 arrivals per period
        >>> result = arrivals.compute()
    """
    def __init__(self, lam):
        self.lam = lam

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm.codes    = np.append(vm.codes,    [_PUSH, _RANDPOISSON])
        vm.operands = np.append(vm.operands, [self.lam, 0])
        return vm


class NegBinom(RandomVariable):
    """Random variable following a negative binomial distribution.
    
    Generates samples from a negative binomial distribution. Models the
    number of successes before r failures occur, with each trial having
    probability p of success.
    
    Args:
        r (float): Number of failures until the experiment is stopped
        p (float): Probability of success in each trial (0 < p < 1)
        
    Example:
        >>> x = NegBinom(5, 0.4)  # r=5 failures, p=0.4 success prob
        >>> result = x.compute()
    """
    def __init__(self, r, p):
        self.r = r
        self.p = p

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm.codes    = np.append(vm.codes,    [_PUSH, _PUSH, _RANDNEGBINOM])
        vm.operands = np.append(vm.operands, [self.r, self.p, 0])
        return vm


class Pert(RandomVariable):
    """Random variable following a PERT (Program Evaluation and Review Technique) distribution.
    
    The PERT distribution is a smooth approximation of a triangular distribution,
    parameterized by minimum, most likely (mode), and maximum values. It is widely
    used in project management and risk analysis for modeling bounded uncertain quantities.
    
    The shape is controlled by a lambda parameter that determines the relative weight
    of the mode compared to the endpoints. Higher lambda values create a sharper
    peak near the mode.
    
    Args:
        low (float): Minimum possible value
        mode (float): Most likely value (must satisfy low < mode < high)
        high (float): Maximum possible value
        lam (float, optional): Shape parameter controlling the peak sharpness.
                              Defaults to 4.0.
        
    Example:
        >>> duration = Pert(3, 5, 10)  # Task duration: min=3, likely=5, max=10
        >>> result = duration.compute()
        >>> result.mean()  # Approximately (3 + 4*5 + 10) / 6 ≈ 5.5
    """
    def __init__(self, low, mode, high, lam=4.0):
        self.low  = low
        self.mode = mode
        self.high = high
        self.lam  = lam

    def _compile(self, vm=None):
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))
        vm.codes    = np.append(vm.codes,    [_PUSH, _PUSH, _PUSH, _PUSH, _RANDPERT])
        vm.operands = np.append(vm.operands, [self.low, self.mode, self.high, self.lam, 0])
        return vm


class DigestVariable(RandomVariable):
    """
    Wrapper for Digest objects to enable use in algebraic operations.

    This class wraps a t-digest data structure in a RandomVariable interface,
    allowing pre-computed distributions to participate in expression trees
    and Monte Carlo simulations alongside parametric distributions.

    The DigestVariable provides access to the underlying t-digest's statistical
    methods (quantile, CDF, bounds) while also enabling the digest to be compiled
    into VM bytecode for sampling operations. When compiled, the digest is converted
    to a histogram representation for efficient random sampling in the VM.

        _digest (Digest): The underlying t-digest data structure

    Example:
        >>> result = x.compute()  # Returns DigestVariable
    """
    def __init__(self, digest: Digest):
        """Initialize with a t-digest.

        Args:
            digest (Digest): The t-digest data structure to wrap
        """
        self._digest = digest

    def quantile(self, q):
        return self._digest.quantile(q)

    def cdf(self, k):
        return self._digest.cdf(k)

    def ccdf(self, k):
        return self._digest.ccdf(k)

    def dcdf(self, k):
        return self._digest.dcdf(k)

    def dccdf(self, k):
        return self._digest.dccdf(k)

    def lower(self):
        """Get the minimum value in the digest.

        Returns:
            int: The smallest value in the digest
        """
        return self._digest.lower()

    def upper(self):
        """Get the maximum value in the digest.

        Returns:
            int: The largest value in the digest
        """
        return self._digest.upper()

    def mean(self):
        return self._digest.mean()

    def _compile(self, vm=None):
        """Compile the digest into VM bytecode for random sampling.

        Converts the t-digest into a histogram representation suitable for
        the virtual machine's random histogram sampling operation. The digest's
        centroids and weights are stored in the VM's histogram arrays and
        referenced by index in the bytecode.

        Steps:
        1. Extracts centroid values and weights from the digest
        2. Stores them in the VM's histogram storage
        3. Emits a RANDHIST opcode referencing the stored histogram
        """
        if vm is None:
            vm = VirtualMachine(np.zeros(1, dtype=np.int64), np.zeros(1))

        # We convert and compile the digest node into a generic histogram for random
        # sampling in the VM
        x = self._digest.getBins()
        w = self._digest.getWeights()
        n = self._digest.getActiveBinCount()

        vm.histBins.append(x)
        vm.histWeights.append(w)
        vm.histN.append(n)

        histIdx = vm.histCount
        vm.histCount += 1

        vm.codes    = np.append(vm.codes,    _RANDHIST)
        vm.operands = np.append(vm.operands, histIdx)

        return vm


class DigestNode(RandomVariable):
    """A random variable node that computes its distribution via Monte Carlo simulation.

    DigestNode wraps another RandomVariable and computes its distribution when
    the compute() method is called, returning a DigestVariable. This is the
    primary mechanism for converting an expression tree into a distribution.

    Args:
        rv (RandomVariable): The expression to evaluate

    Example:
        >>> x = Normal(0, 1)
        >>> node = DigestNode(x)
        >>> result = node.compute(samples=10000)
    """
    def __init__(self, rv):
        self.rv = rv

    def compute(self, samples=10000, maxBins=32):
        """Evaluate the expression tree via Monte Carlo simulation.

        Compiles the expression tree into VM bytecode and runs it for the
        specified number of samples, collecting results into a t-digest.

        Args:
            samples (int, optional): Number of Monte Carlo samples. Defaults to 10000.
            maxBins (int, optional): Maximum bins for the t-digest. Defaults to 32.

        Returns:
            DigestVariable: A wrapper around the t-digest containing the
                           simulated distribution
        """
        vm     = self.rv._compile()
        vm     = vm
        digest =  vm.compute(samples=samples, maxBins=maxBins)
        return DigestVariable(digest)

    def _compile(self, vm=None):
        return self.rv._compile(vm)
