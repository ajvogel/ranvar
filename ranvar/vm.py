import numpy as np
import cython as pyx

from .digest import Digest

if pyx.compiled:
    from cython.cimports.libc.math import ceil as c_ceil
    from cython.cimports.libc.math import floor as c_floor
    from cython.cimports.libc.math import log as c_log
    from cython.cimports.libc.math import sqrt as c_sqrt
    from cython.cimports.libc.math import exp as c_exp
    from cython.cimports.libc.stdlib import rand as c_rand
    from cython.cimports.libc.stdlib import RAND_MAX as c_RAND_MAX
else:
    pass


#---[ Probability Distributions ]-----------------------------------------------
# These functions are duplicated from digest.py because @pyx.cfunc functions
# cannot be imported across Cython modules without complex pxd file setups.

# Rand
@pyx.cfunc
@pyx.cdivision(True)
def _rand() -> pyx.double:
    out:pyx.double = pyx.cast(pyx.double, c_rand()) / pyx.cast(pyx.double, c_RAND_MAX)
    return out

# RandInt
@pyx.cfunc
def _randint(l: pyx.double, h: pyx.double) -> pyx.double:
    l2: pyx.double  = l - 1
    out: pyx.double = c_ceil((h - l2) * _rand() + l2)

    if out < l:
        out = l

    return out

@pyx.cfunc
def _randnorm(mu: pyx.double, stdev: pyx.double) -> pyx.double:
    """
    https://en.wikipedia.org/wiki/Marsaglia_polar_method
    """
    while True:
        x:pyx.double = 2*_rand() - 1
        y:pyx.double = 2*_rand() - 1

        s: pyx.double = x**2 + y**2

        if s < 1:
            break

    z: pyx.double = x * c_sqrt( -2*c_log(s) / s)
    return z * stdev + mu


@pyx.cfunc
@pyx.cdivision(True)
def _randexp(rate: pyx.double) -> pyx.double:
    """
    Sample from exponential distribution with given rate parameter.
    Uses inverse transform sampling: X = -ln(U) / rate
    """
    u: pyx.double = _rand()
    # Avoid log(0)
    while u == 0:
        u = _rand()
    return -c_log(u) / rate


@pyx.cfunc
@pyx.cdivision(True)
def _randgamma(shape: pyx.double, scale: pyx.double) -> pyx.double:
    """
    Sample from gamma distribution with given shape (k) and scale (theta) parameters.
    Uses Marsaglia and Tsang's method for shape >= 1, and Ahrens-Dieter method for shape < 1.
    https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables
    """
    d: pyx.double
    c: pyx.double
    x: pyx.double
    v: pyx.double
    u: pyx.double
    shape_use: pyx.double = shape

    # For shape < 1, use the transformation: if X ~ Gamma(shape+1), then X * U^(1/shape) ~ Gamma(shape)
    if shape < 1:
        shape_use = shape + 1

    # Marsaglia and Tsang's method for shape >= 1
    d = shape_use - 1.0 / 3.0
    c = 1.0 / c_sqrt(9.0 * d)

    while True:
        while True:
            x = _randnorm(0, 1)
            v = 1.0 + c * x
            if v > 0:
                break

        v = v * v * v
        u = _rand()

        if u < 1.0 - 0.0331 * (x * x) * (x * x):
            break

        if c_log(u) < 0.5 * x * x + d * (1.0 - v + c_log(v)):
            break

    result: pyx.double = d * v * scale

    # Transform back if original shape < 1
    # U must be strictly in (0, 1) for the Ahrens-Dieter method.
    # U == 0 causes log(0) = -inf, and U == 1 causes U^(1/shape) = 1
    # which skips the scaling entirely, returning Gamma(shape+1) instead of Gamma(shape).
    if shape < 1:
        u = _rand()
        while u <= 0 or u >= 1:
            u = _rand()
        result = result * c_exp(c_log(u) / shape)

    return result


@pyx.cfunc
@pyx.cdivision(True)
def _randpoisson(lam: pyx.double) -> pyx.double:
    """
    Sample from Poisson distribution with given rate (lambda) parameter.
    Uses Knuth's algorithm for small lambda, normal approximation for large lambda.
    """
    k: pyx.int
    p: pyx.double
    L: pyx.double
    u: pyx.double
    result: pyx.double

    if lam <= 0:
        return 0.0

    if lam < 30:
        # Knuth's algorithm for small lambda
        L = c_exp(-lam)
        k = 0
        p = 1.0

        while True:
            k = k + 1
            u = _rand()
            p = p * u
            if p <= L:
                break

        result = pyx.cast(pyx.double, k - 1)
    else:
        # Normal approximation for large lambda
        result = _randnorm(lam, c_sqrt(lam))
        if result < 0:
            result = 0
        result = c_floor(result + 0.5)

    return result


@pyx.cfunc
@pyx.cdivision(True)
def _randnegbinom(n: pyx.double, p: pyx.double) -> pyx.double:
    """
    Sample from negative binomial distribution with parameters n and p.
    Uses the Gamma-Poisson mixture representation:
    If Y ~ Gamma(n, (1-p)/p) and X | Y ~ Poisson(Y), then X ~ NegativeBinomial(n, p)
    """
    # Sample from Gamma(n, (1-p)/p)
    # scale = (1-p)/p
    scale: pyx.double = (1.0 - p) / p
    gamma_sample: pyx.double = _randgamma(n, scale)

    # Sample from Poisson(gamma_sample)
    return _randpoisson(gamma_sample)


#======================================[ Virtual Machine ]=========================================
# pyx.declare creates c constants.
_PASS = pyx.declare(pyx.int, 0)
_PUSH = pyx.declare(pyx.int, 1)

_DROP  = pyx.declare(pyx.int, 2)
_STORE = pyx.declare(pyx.int, 3)
_LOAD  = pyx.declare(pyx.int, 4)

# Onetary Ops
_NEG = pyx.declare(pyx.int, 10)
_ABS = pyx.declare(pyx.int, 11)

# Binary Ops
_ADD = pyx.declare(pyx.int, 20)
_MUL = pyx.declare(pyx.int, 21)
_POW = pyx.declare(pyx.int, 22)
_DIV = pyx.declare(pyx.int, 23)
_SUB = pyx.declare(pyx.int, 24)
_MOD = pyx.declare(pyx.int, 25)
_FLOORDIV = pyx.declare(pyx.int, 26)
_LT  = pyx.declare(pyx.int, 27)
_LE  = pyx.declare(pyx.int, 28)
_MAX = pyx.declare(pyx.int, 29)
_MIN = pyx.declare(pyx.int, 29)

_BINOPMAX = pyx.declare(pyx.int, 50)

# Summation Thingies...
#OP_SUM_START:pyx.int = 51
_SUM_START = pyx.declare(pyx.int, 51)
_SUM_END   = pyx.declare(pyx.int, 52)

# Statistical Ops
_RANDINT = pyx.declare(pyx.int, 100)
_RANDNORM = pyx.declare(pyx.int, 101)
_RAND_QUANTILES = pyx.declare(pyx.int, 102)
_ARRAY_SUM = pyx.declare(pyx.int, 103)
_RAND_HIST = pyx.declare(pyx.int, 104)
_RAND_NEGBINOM = pyx.declare(pyx.int, 105)

@pyx.cclass
class VirtualMachine():
    """
    A stack-based virtual machine for executing Monte Carlo simulation programs.

    This virtual machine implements a custom instruction set designed for efficient
    execution of Monte Carlo simulations. It uses a stack-based architecture with
    support for variables, loops, random number generation, and statistical operations.

    The VM supports the following major features:
    - Stack-based arithmetic operations (add, multiply, divide, etc.)
    - Variable storage and retrieval (26 variables: a-z)
    - Looping constructs with sum aggregation
    - Random number generation (uniform, normal, integers)
    - Statistical sampling from quantiles and histograms
    - Array operations and summation

    Attributes:
        codes (np.ndarray): Array of operation codes to execute
        operands (np.ndarray): Array of operands corresponding to each operation
        stack (np.ndarray): Execution stack for intermediate values
        variables (np.ndarray): Storage for 26 variables (indexed 0-25 for a-z)
        pointers (np.ndarray): Stack for storing loop return addresses
        iterators (np.ndarray): Stack for storing loop iteration counts
        stackCount (int): Current number of items on the execution stack
        counter (int): Current instruction pointer
        pointerCount (int): Current number of items on the pointer stack
        iterCount (int): Current number of items on the iterator stack

    Example:
        >>> codes = np.array([_PUSH, _PUSH, _ADD])
        >>> operands = np.array([5.0, 3.0, 0.0])
        >>> vm = VirtualMachine(codes, operands)
        >>> result = vm.sample()  # Returns 8.0
    """

    # Core Op Codes


    _codes: pyx.double[:]
    _operands: pyx.double[:]
    _stack: pyx.double[:]
    _variables: pyx.double[:]
    _pointers: pyx.long[:]
    _iterators: pyx.double[:]

    codes: np.ndarray
    operands: np.ndarray
    stack: np.ndarray
    variables: np.ndarray
    stackCount: pyx.int
    counter: pyx.int
    pointerCount: pyx.int
    pointers: np.ndarray
    iterCount: pyx.int
    iterators: np.ndarray
    def __init__(self, codes, operands) -> None:
        """Initialize the virtual machine with program code and operands.

        Args:
            codes (np.ndarray): Array of operation codes defining the program
            operands (np.ndarray): Array of operands for each operation

        Note:
            The codes and operands arrays must have the same length.
        """
        self.codes    = codes
        self.operands = operands
        self.stack    = np.zeros(100)
        self.variables = np.zeros(26)
        self.pointers = np.zeros(16, dtype=np.int_)
        self.iterators = np.zeros(16)
        self.stackCount = 0
        self.pointerCount = 0
        self.iterCount = 0
        self.counter = 0


        # Init the memory view.fdd
        self._codes = self.codes
        self._operands = self.operands
        self._stack   = self.stack
        self._variables = self.variables
        self._pointers = self.pointers
        self._iterators = self.iterators

    def reset(self):
        return self._reset()

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    def _reset(self) -> pyx.void:
        """Reset the virtual machine state for a new execution.

        Clears all stacks, resets counters, and reinitializes memory arrays.
        This allows the same VM instance to be used for multiple executions.
        """
        self.stackCount   = 0
        self.iterCount    = 0
        self.pointerCount = 0
        self.counter      = 0

        # self.stack    = np.zeros(100)
        # self.variables = np.zeros(26)
        # self.pointers = np.zeros(16, dtype=np.int_)
        # self.iterators = np.zeros(16)

        i: pyx.int
        for i in range(100):
            self._stack[i] = 0

        for i in range(26):
            self._variables[i] = 0

        for i in range(16):
            self._pointers[i]  = 0
            self._iterators[i] = 0



    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def pushPointer(self, value: pyx.int) -> pyx.void:
        """Push a pointer (return address) onto the pointer stack.

        Used for implementing loop constructs by storing the instruction
        address to return to after loop completion.

        Args:
            value (int): The instruction pointer value to store
        """
        self._pointers[self.pointerCount] = value
        self.pointerCount += 1

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popPointer(self) -> pyx.int:
        """Pop a pointer (return address) from the pointer stack.

        Returns:
            int: The most recently pushed pointer value

        Raises:
            AssertionError: If the pointer stack is empty
        """
        assert self.pointerCount > 0
        self.pointerCount -= 1
        return self._pointers[self.pointerCount]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def pushIterator(self, value: pyx.double) -> pyx.void:
        """Push an iterator count onto the iterator stack.

        Used for implementing loop constructs by tracking remaining
        iterations for nested loops.

        Args:
            value (float): The iteration count to store
        """
        self._iterators[self.iterCount] = value
        self.iterCount += 1

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popIterator(self) -> pyx.double:
        """Pop an iterator count from the iterator stack.

        Returns:
            float: The most recently pushed iterator count

        Raises:
            AssertionError: If the iterator stack is empty
        """
        assert self.iterCount > 0
        self.iterCount -= 1
        return self._iterators[self.iterCount]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def peekIterator(self) -> pyx.double:
        """Return the top iterator count without removing it from the stack.

        Returns:
            float: The top iterator count value
        """
        return self._iterators[self.iterCount - 1]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def peekPointer(self) -> pyx.int:
        """Return the top pointer without removing it from the stack.

        Returns:
            int: The top pointer value
        """
        return self._pointers[self.pointerCount - 1]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def pushStack(self, value: pyx.double):
        """Push a value onto the execution stack.

        Args:
            value (float): The value to push onto the stack
        """
        self._stack[self.stackCount] = value
        self.stackCount += 1

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popStack(self) -> pyx.float:
        """Pop a value from the execution stack.

        Returns:
            float: The most recently pushed value

        Raises:
            AssertionError: If the execution stack is empty
        """
        # assert self.stackCount > 0
        self.stackCount -= 1
        return self._stack[self.stackCount]

    @pyx.cfunc
    def _dropStack(self, cnt:pyx.int = 1):
        """Drop (discard) multiple values from the execution stack.

        Args:
            cnt (int, optional): Number of values to drop. Defaults to 1.
        """
        i: pyx.int
        #print('Dropping ',cnt, ' values...')
        for i in range(cnt):
            #print('    Drop', i, '...')
            self.popStack()

    @pyx.cfunc
    def _store(self, varNumber: pyx.double) -> pyx.void:
        """Store the top stack value into a variable.

        Pops a value from the execution stack and stores it in the specified
        variable slot (0-25 corresponding to variables a-z).

        Args:
            varNumber (float): Variable index (0-25) to store the value in
        """
        idx: pyx.int = pyx.cast(pyx.int, varNumber)
        varValue = self.popStack()
        self._variables[idx] = varValue

    @pyx.cfunc
    def _load(self, varNumber: pyx.double) -> pyx.void:
        """Load a variable value onto the execution stack.

        Pushes the value from the specified variable slot onto the stack.

        Args:
            varNumber (float): Variable index (0-25) to load the value from
        """
        idx: pyx.int = pyx.cast(pyx.int, varNumber)
        self.pushStack(self._variables[idx])

    @pyx.cfunc
    def _sumStart(self, loopNumber: pyx.double) -> pyx.void:
        """Initialize a summation loop.

        Sets up the loop state by:
        1. Popping the number of iterations from the stack
        2. Pushing an initial sum of 0 onto the stack
        3. Storing the iteration count and return address

        Args:
            loopNumber (float): Loop identifier (currently unused)
        """
        idx: pyx.int = pyx.cast(pyx.int, loopNumber)
        nTerms = self.popStack()
        self.pushStack(0)
        self.pushIterator(nTerms)
        self.pushPointer(self.counter)

    @pyx.cfunc
    def _sumEnd(self, loopNumber: pyx.double) -> pyx.void:
        """Process one iteration of a summation loop.

        Adds the current value to the running sum, decrements the iteration
        counter, and either continues the loop or exits if complete.

        Args:
            loopNumber (float): Loop identifier (currently unused)
        """

        idx: pyx.int = pyx.cast(pyx.int, loopNumber)

        # First we add the running total to the answer.
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 + x2)
        self.pushIterator(self.popIterator() - 1)

        if self.peekIterator() > 0:
            self.counter = self.peekPointer()

        else:
            self.popPointer()
            self.popIterator()



    @pyx.cfunc
    def _binop(self, opCode: pyx.double) -> pyx.void:
        """Execute a binary operation on the top two stack values.

        Pops two values from the stack (x1, x2), performs the specified
        operation, and pushes the result back onto the stack.

        Supported operations:
        - ADD: x1 + x2
        - MUL: x1 * x2
        - POW: x1 ** x2
        - DIV: x1 / x2
        - FLOORDIV: x1 // x2
        - MOD: x1 % x2
        - SUB: x1 - x2
        - LT: 1 if x1 < x2 else 0
        - LE: 1 if x1 <= x2 else 0

        Args:
            opCode (float): Operation code specifying which operation to perform
        """
        # When things are removed from the stack we pop them from the bottom of the stack
        # in the reverse order in which they were pushed. So we pop x2 first before x1.


        x2 = self.popStack()
        x1 = self.popStack()
        if opCode == _ADD:
            self.pushStack(x1 + x2)
        elif opCode == _MUL:
            self.pushStack(x1 * x2)
        elif opCode == _POW:
            self.pushStack(x1 ** x2)
        elif opCode == _DIV:
            self.pushStack(x1 / x2)
        elif opCode == _FLOORDIV:
            self.pushStack(x1 // x2)
        elif opCode == _MOD:
            self.pushStack(x1 % x2)
        elif opCode == _SUB:
            self.pushStack(x1 - x2)
        elif opCode == _LT:
            self.pushStack(1) if x1 < x2 else self.pushStack(0)
        elif opCode == _LE:
            self.pushStack(1) if x1 <= x2 else self.pushStack(0)
        elif opCode == _MAX:
            self.pushStack(x1) if x1 > x2 else self.pushStack(x2)
        elif opCode == _MIN:
            self.pushStack(x1) if x1 < x2 else self.pushStack(x2)

    @pyx.cfunc
    def _randInt(self) -> pyx.void:
        """Generate a random integer within a specified range.

        Pops two values from the stack (low, high) and pushes a random
        integer in the range [low, high] inclusive.
        """
        h = self.popStack()
        l = self.popStack()

        # x: pyx.double = c_round((h - l)*_rand() + l)
        self.pushStack(_randint(l, h))
        #self.pushStack(pyx.cast(pyx.double, _randint(l,h)))
        #self.pushStack(random.randint(int(l),int(h)))

    @pyx.cfunc
    def _randNorm(self) -> pyx.void:
        """Generate a random number from a normal distribution.

        Pops two values from the stack (mean, std_dev) and pushes a random
        number drawn from the specified normal distribution.
        """
        std: pyx.double = self.popStack()
        mu: pyx.double    = self.popStack()

        self.pushStack(_randnorm(mu, std))

    @pyx.cfunc
    def _randNegBinom(self) -> pyx.void:
        """Generate a random number from a negative binomial distribution.

        Pops two values from the stack (n, p) and pushes a random
        number drawn from the negative binomial distribution.
        n is the number of successes, p is the probability of success.
        """
        p: pyx.double = self.popStack()
        n: pyx.double = self.popStack()

        self.pushStack(_randnegbinom(n, p))

    @pyx.cfunc
    def _arraySum(self, nArray: pyx.double) -> pyx.void:
        """Sum elements from an array within a specified range.

        Pops start and end indices, then pops nArray values from the stack.
        Sums only those values whose indices fall within [start, end) and
        pushes the result.

        Args:
            nArray (float): Number of array elements to pop from the stack
        """
        som: pyx.double = 0.0
        i: pyx.int
        nArrayInt: pyx.int = pyx.cast(pyx.int, nArray)

        start: pyx.double = self.popStack()
        end:pyx.double    = self.popStack()

        for i in range(nArrayInt):
            x: pyx.double = self.popStack()
            if (start <= i) and (i < end):
                som += x

        self.pushStack(som)

    @pyx.cfunc
    def _randQuantiles(self, nBins: pyx.double) -> pyx.void:
        """Sample from a distribution defined by quantile values.

        Expects nBins quantile values on the stack representing a cumulative
        distribution. Generates a random sample by linear interpolation
        between adjacent quantiles.

        The method:
        1. Generates a random uniform value
        2. Finds the quantile interval containing this value
        3. Interpolates linearly between the bounding quantiles
        4. Pushes the interpolated result

        Args:
            nBins (float): Number of quantile bins on the stack

        Example:
            For quantiles [0.1, 0.15, 0.29, 0.43, 0.57, 0.71, 0.85, 0.99]
            corresponding to values [x0, x1, x2, x3, x4, x5, x6, x7]
        """
        # There is one fewer intervals than there are actual points.
        dY: pyx.double = 1. / (nBins - 1)
        y_: pyx.double = _rand()
        #y_: pyx.double = 0.55


        i: pyx.double = c_floor(y_ / dY)

        #print(i)
        #print('Stack Count:', self.stackCount)

        self._dropStack(pyx.cast(pyx.int, i))

        #print('Stack Count after initial drop:', self.stackCount)

        xi:pyx.double = self.popStack()
        xi_n:pyx.double = self.popStack()

        yi:pyx.double   = i*dY
        yi_n:pyx.double = yi + dY

        #print(xi, ' -> ', xi_n)

        self._dropStack(pyx.cast(pyx.int, nBins - i - 2))

        #print('Stack Count after dropping remaining:', self.stackCount)

        m:pyx.double = (xi_n - xi) / (yi_n - yi)

        x_:pyx.double = xi + m*(y_ - yi)

        self.pushStack(x_)


    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    @pyx.linetrace(True)
    def _randHist(self, nBins: pyx.double) -> pyx.void:
        """Sample from a histogram representation.

        Expects pairs of (value, cumulative_probability) on the stack
        representing a histogram. Generates a random sample by finding
        the appropriate bin and interpolating within it.

        The method:
        1. Generates a random uniform value p
        2. Finds the histogram bin where cumulative probability brackets p
        3. Linearly interpolates the value within that bin
        4. Pushes the interpolated result

        Args:
            nBins (float): Number of histogram bins

        Note:
            Stack should contain alternating (xi, ci) pairs where xi is the
            bin value and ci is the cumulative probability.
        """
        p:    pyx.double
        xi:   pyx.double
        ci:   pyx.double
        xi_n: pyx.double
        ci_n: pyx.double
        m:    pyx.double
        x_:   pyx.double
        i_n:  pyx.int
        nB:   pyx.int

        nB = pyx.cast(pyx.int, nBins)

        p = _rand()
        #p = 0.25

        # print("p = ", p)

        xi = self.popStack()
        ci = self.popStack()

        # print("xi = ", xi)
        # print('ci = ', ci)

        x_ = 0

        for i_n in range(1, nB):
            # print("i_n")
            xi_n = self.popStack()
            ci_n = self.popStack()

            # print("xi_n = ", xi)
            # print('ci_n = ', ci)

            # print(ci, "<=", p,"<", ci_n, "  ",ci <= p < ci_n)

            if ci <= p < ci_n:
                m  = (xi_n - xi) / (ci_n - ci)
                x_ = xi + m*(p - ci)
                # print('x* = ', x_)

            elif (i_n == nBins-1) and (p == 1):
                # If this is the last bin and p is exactly 1 we will miss the last value
                # adding a second check for it here.
                x_ = xi_n

            xi = xi_n
            ci = ci_n


        #print('x* = ', x_)
        self.pushStack(x_)





    def printState(self):
        """Print the current state of the virtual machine for debugging.

        Displays the current instruction, operand, stack contents, and
        pointer stack state. Useful for debugging VM execution.
        """
        _stack = []
        for i in reversed(range(self.stackCount)):
            _stack.append(self.stack[i])

        _stack = " ".join([f'{s:.0f}' for s in _stack ])

        print(f'{self.counter}: {self._codes[self.counter]:.0f}     {self._operands[self.counter]} -> [{_stack}]    {self.pointers[:2]}')





    @pyx.ccall
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def sample(self) -> pyx.float:
        """Execute the loaded program and return the final result.

        Runs the virtual machine by executing all instructions in sequence.
        The program should leave exactly one value on the stack, which is
        returned as the result.

        Returns:
            float: The final value from the execution stack

        Note:
            The VM state (stack, variables, counters) is modified during
            execution. Call reset() before sample() for clean execution.
        """

        # for e, (c, o) in enumerate(list(zip(self.codes, self.operands))):
        #     print(f'{e}: {c}   {o}')

        N:pyx.int = self._codes.shape[0]
        #i:pyx.int = 0
        opCode: pyx.double
        operand: pyx.double

        while self.counter < N:
            #self.printState()
            opCode = self._codes[self.counter]
            operand = self._operands[self.counter]

            if   opCode == _PASS:
                pass
            elif opCode == _PUSH:
                self.pushStack(self._operands[self.counter])
            elif opCode == _STORE:
                self._store(self._operands[self.counter])
            elif opCode == _LOAD:
                self._load(self._operands[self.counter])
            elif opCode == _SUM_START:
                self._sumStart(self._operands[self.counter])
            elif opCode == _SUM_END:
                self._sumEnd(self._operands[self.counter])
            elif _ADD <= opCode <= _BINOPMAX:
                self._binop(opCode)

            elif opCode == _RANDINT:
                self._randInt()
            elif opCode == _RANDNORM:
                self._randNorm()
            elif opCode == _RAND_QUANTILES:
                self._randQuantiles(operand)
            elif opCode == _ARRAY_SUM:
                self._arraySum(operand)
            elif opCode == _RAND_HIST:
                self._randHist(operand)
            elif opCode == _RAND_NEGBINOM:
                self._randNegBinom()

            self.counter += 1

        return self.popStack()

    @pyx.linetrace(True)
    def compute(self, samples:pyx.int=10000, maxBins:pyx.int=32):
        """Run the simulation multiple times and collect results in a t-digest.

        Executes the VM program for the specified number of samples, resetting
        the VM state between each execution. Results are accumulated in a t-digest
        for efficient quantile computation.

        Args:
            samples (int, optional): Number of simulation runs. Defaults to 10000.
            maxBins (int, optional): Maximum bins for the t-digest. Defaults to 32.

        Returns:
            Digest: A t-digest containing the distribution of simulation results
        """
        rv = Digest(maxBins=maxBins)
        i: pyx.int
        for i in range(samples):
            x:pyx.float = self.sample()
            self._reset()
            rv._add(x, 1)

        return rv

    def run(self):
        """Run the simulation with default parameters.

        Convenience method that calls compute() with default parameters.

        Returns:
            Digest: A t-digest containing the simulation results
        """
        return self.compute()

#==================================================================================================
