import numpy as np
import cython as pyx

from .digest_cpp import Digest

if pyx.compiled:
    from cython.cimports.libc.math import floor as c_floor
    from cython.cimports.ranvar.random import rand, randint, randnorm, randexp, randgamma, randpoisson, randnegbinom, randpert
else:
    pass


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
_RAND_GAMMA = pyx.declare(pyx.int, 106)
_RAND_PERT = pyx.declare(pyx.int, 107)

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
        """
        Initialize the virtual machine with program code and operands.

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
        """
        Reset the virtual machine state for a new execution.

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
        """
        Push a pointer (return address) onto the pointer stack.

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
        """
        Pop a pointer (return address) from the pointer stack.

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
        """
        Push an iterator count onto the iterator stack.

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
        """
        Pop an iterator count from the iterator stack.

        Returns:
            float: The most recently pushed iterator value

        Raises:
            AssertionError: If the iterator stack is empty
        """
        assert self.iterCount > 0
        self.iterCount -= 1
        return self._iterators[self.iterCount]


    @pyx.ccall
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def pushStack(self, value: pyx.double) -> pyx.void:
        """
        Push a value onto the execution stack.

        Args:
            value (float): The value to push onto the stack
        """
        self._stack[self.stackCount] = value
        self.stackCount += 1

    @pyx.ccall
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popStack(self) -> pyx.double:
        """
        Pop a value from the execution stack.

        Returns:
            float: The top value on the stack

        Raises:
            AssertionError: If the stack is empty
        """
        assert self.stackCount > 0
        self.stackCount -= 1
        return self._stack[self.stackCount]


    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _add(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 + x2)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _sub(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x2 - x1)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _mul(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 * x2)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _div(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x2 / x1)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _mod(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x2 % x1)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    def _floordiv(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(pyx.cast(pyx.double, pyx.cast(pyx.long, x2) // pyx.cast(pyx.long, x1)))

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _pow(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x2 ** x1)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _neg(self) -> pyx.void:
        x1 = self.popStack()
        self.pushStack(-x1)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _abs(self) -> pyx.void:
        x1 = self.popStack()
        if x1 < 0:
            self.pushStack(-x1)
        else:
            self.pushStack(x1)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _lt(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x2 < x1:
            self.pushStack(1.0)
        else:
            self.pushStack(0.0)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _le(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x2 <= x1:
            self.pushStack(1.0)
        else:
            self.pushStack(0.0)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _max(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x1 > x2:
            self.pushStack(x1)
        else:
            self.pushStack(x2)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _min(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x1 < x2:
            self.pushStack(x1)
        else:
            self.pushStack(x2)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    def _randInt(self) -> pyx.void:
        h = self.popStack()
        l = self.popStack()
        self.pushStack(pyx.cast(pyx.double, randint(l, h)))

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _randNorm(self) -> pyx.void:
        stdev = self.popStack()
        mu    = self.popStack()
        self.pushStack(randnorm(mu, stdev))

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _randExp(self) -> pyx.void:
        lam = self.popStack()
        self.pushStack(randexp(lam))

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _randGamma(self) -> pyx.void:
        loc   = self.popStack()
        theta = self.popStack()
        k     = self.popStack()
        self.pushStack(randgamma(k, theta) + loc)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _randPoisson(self) -> pyx.void:
        lam = self.popStack()
        self.pushStack(pyx.cast(pyx.double, randpoisson(lam)))

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _randNegBinom(self) -> pyx.void:
        p = self.popStack()
        r = self.popStack()
        self.pushStack(pyx.cast(pyx.double, randnegbinom(r, p)))

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def _randPert(self) -> pyx.void:
        lam  = self.popStack()
        high = self.popStack()
        mode = self.popStack()
        low  = self.popStack()
        self.pushStack(randpert(low, mode, high))

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    def _randQuantiles(self, n: pyx.int) -> pyx.void:
        """
        Sample a random value from a quantile distribution stored in operands.

        This method implements inverse transform sampling using quantiles
        stored in the operands array. It generates a uniform random number
        and maps it to the corresponding quantile value through linear
        interpolation.

        Args:
            n (int): The number of quantile values in the distribution
        """
        u: pyx.double = rand()
        N: pyx.int = n

        # i == index of operands we're on right now.
        i: pyx.int = self.counter + 1

        # the index * (N-1) at the bounds [0, N-1] is a float telling us our position in the
        # quantile array.
        # At the lower bound (u==0), we want index 0.
        # At the upper bound (u==1), we want index N-1.
        fl_index: pyx.double = u * (N - 1)

        # Lower and upper bounds of interval
        idx_low: pyx.int  = pyx.cast(pyx.int, c_floor(fl_index))
        idx_high: pyx.int = idx_low + 1

        # Avoid out of bound access when u==1
        if idx_high > N-1:
            idx_high = N-1

        # The fraction tells us how far between the two bounds we are
        frac: pyx.double = fl_index - pyx.cast(pyx.double, idx_low)

        # Interpolate the result
        val_low  = self._operands[i + idx_low]
        val_high = self._operands[i + idx_high]

        result: pyx.double = val_low + frac * (val_high - val_low)

        # Advance the instruction pointer past the data block
        self.counter += N

        self.pushStack(result)

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    def _randHist(self, n: pyx.int) -> pyx.void:
        """
        Sample a random value from a histogram distribution stored in operands.

        Uses inverse CDF sampling with a piecewise uniform distribution defined
        by the histogram bins and weights stored in the operands array.

        The histogram data is stored as interleaved (bin, weight) pairs in the
        operands array, with n pairs total.

        Args:
            n (int): The number of bins in the histogram
        """
        u: pyx.double = rand()
        N: pyx.int = n

        # i == base index in operands for this histogram
        base: pyx.int = self.counter + 1

        # Calculate total weight
        total_weight: pyx.double = 0.0
        j: pyx.int
        for j in range(N):
            total_weight += self._operands[base + j*2 + 1]

        # Find the bin
        cumulative: pyx.double = 0.0
        threshold: pyx.double  = u * total_weight
        result: pyx.double = self._operands[base]  # default to first bin

        for j in range(N):
            cumulative += self._operands[base + j*2 + 1]
            if cumulative >= threshold:
                result = self._operands[base + j*2]
                break

        # Advance past histogram data (N bins * 2 doubles each)
        self.counter += N * 2

        self.pushStack(result)


    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    def _arraySum(self, n: pyx.int) -> pyx.void:
        """
        Pop n values from the stack and push their sum.

        Args:
            n (int): The number of values to sum
        """
        result: pyx.double = 0.0
        i: pyx.int
        for i in range(n):
            result += self.popStack()
        self.pushStack(result)


    @pyx.ccall
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.cdivision(True)
    def sample(self) -> pyx.double:
        """
        Execute the VM program once and return the result.

        Runs the compiled bytecode from start to finish, using any random
        sampling instructions to draw a single sample from the distribution.

        Returns:
            float: The result of executing the VM program once.
        """
        N: pyx.int = self._codes.shape[0]
        opCode: pyx.double
        n: pyx.int

        while self.counter < N:
            opCode = self._codes[self.counter]

            if   opCode == _PASS:
                pass
            elif opCode == _PUSH:
                self.pushStack(self._operands[self.counter])
            elif opCode == _DROP:
                self.popStack()
            elif opCode == _STORE:
                slot = pyx.cast(pyx.int, self._operands[self.counter])
                self._variables[slot] = self.popStack()
            elif opCode == _LOAD:
                slot = pyx.cast(pyx.int, self._operands[self.counter])
                self.pushStack(self._variables[slot])
            elif opCode == _NEG:
                self._neg()
            elif opCode == _ABS:
                self._abs()
            elif opCode == _ADD:
                self._add()
            elif opCode == _SUB:
                self._sub()
            elif opCode == _MUL:
                self._mul()
            elif opCode == _DIV:
                self._div()
            elif opCode == _MOD:
                self._mod()
            elif opCode == _FLOORDIV:
                self._floordiv()
            elif opCode == _POW:
                self._pow()
            elif opCode == _LT:
                self._lt()
            elif opCode == _LE:
                self._le()
            elif opCode == _MAX:
                self._max()
            elif opCode == _MIN:
                self._min()
            elif opCode == _RANDINT:
                self._randInt()
            elif opCode == _RANDNORM:
                self._randNorm()
            elif opCode == _RAND_NEGBINOM:
                self._randNegBinom()
            elif opCode == _RAND_GAMMA:
                self._randGamma()
            elif opCode == _RAND_PERT:
                self._randPert()
            elif opCode == _RAND_QUANTILES:
                n = pyx.cast(pyx.int, self._operands[self.counter])
                self._randQuantiles(n)
            elif opCode == _RAND_HIST:
                n = pyx.cast(pyx.int, self._operands[self.counter])
                self._randHist(n)
            elif opCode == _ARRAY_SUM:
                n = pyx.cast(pyx.int, self._operands[self.counter])
                self._arraySum(n)
            elif opCode == _SUM_START:
                self.pushIterator(self._operands[self.counter])
                self.pushPointer(self.counter)
            elif opCode == _SUM_END:
                count = self.popIterator()
                if count > 1:
                    self.pushIterator(count - 1)
                    ptr = self.popPointer()
                    self.pushPointer(ptr)
                    self._add()
                    self.counter = ptr
                else:
                    self.popPointer()

            self.counter += 1

        return self.popStack()

    def compute(self, samples:pyx.int=10_000, maxBins:pyx.int=32):
        """
        Run the simulation multiple times and collect results in a t-digest.

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
            rv.add(x, 1)

        return rv

    def run(self):
        """
        Run the simulation with default parameters.

        Convenience method that calls compute() with default parameters.

        Returns:
            Digest: A t-digest containing the simulation results
        """
        return self.compute()

#==================================================================================================
