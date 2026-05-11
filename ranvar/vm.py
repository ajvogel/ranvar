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
_MOD = pyx.declare(pyx.int, 24)
_FLOORDIV = pyx.declare(pyx.int, 25)
_SUB = pyx.declare(pyx.int, 26)
_MAX = pyx.declare(pyx.int, 27)
_MIN = pyx.declare(pyx.int, 28)
_GT  = pyx.declare(pyx.int, 29)
_GTE = pyx.declare(pyx.int, 30)
_LT  = pyx.declare(pyx.int, 31)
_LTE = pyx.declare(pyx.int, 32)

# Sampling Ops
_RANDINT     = pyx.declare(pyx.int, 40)
_RANDNORM    = pyx.declare(pyx.int, 41)
_RANDEXP     = pyx.declare(pyx.int, 42)
_RANDGAMMA   = pyx.declare(pyx.int, 43)
_RANDPOISSON = pyx.declare(pyx.int, 44)
_RANDNEGBINOM = pyx.declare(pyx.int, 45)
_RANDPERT     = pyx.declare(pyx.int, 46)

# Histogram Op
_RANDHIST = pyx.declare(pyx.int, 50)


@pyx.cclass
class VirtualMachine():
    """Stack-based virtual machine for executing random variable expressions.

    The VirtualMachine (VM) is the core execution engine for the ranvar DSL.
    It evaluates expression trees compiled into bytecode instructions using a
    stack-based architecture. Each instruction either pushes values onto the
    stack, performs arithmetic operations, or generates random samples.

    The VM supports:
    - Basic arithmetic: +, -, *, /, //, %, **
    - Comparison operators: <, <=, >, >=
    - Random distributions: uniform integer, normal, exponential, gamma,
      Poisson, negative binomial, PERT
    - Histogram sampling from empirical distributions

    Attributes:
        codes (np.ndarray): Array of integer opcodes
        operands (np.ndarray): Array of floating-point operand values
        stack (np.ndarray): Execution stack for intermediate values
        stackSize (int): Maximum stack depth
        stackCount (int): Current stack pointer
        memory (np.ndarray): Named variable storage (STORE/LOAD ops)
        memoryCount (int): Number of allocated memory slots
        histBins (list): Centroid values for histogram distributions
        histWeights (list): Weights for histogram distributions
        histN (list): Active bin counts for histogram distributions
        histCount (int): Number of histogram distributions stored

    Example:
        >>> from ranvar import Normal
        >>> x = Normal(0, 1)
        >>> vm = x._compile()
        >>> result = vm.compute(samples=10000)
        >>> result.quantile(0.5)  # approximately 0
    """
    codes:    np.ndarray
    operands: np.ndarray
    stack:    np.ndarray
    memory:   np.ndarray

    _codes:    pyx.long[:]
    _operands: pyx.double[:]
    _stack:    pyx.double[:]
    _memory:   pyx.double[:]

    stackSize:  pyx.int
    stackCount: pyx.int
    memoryCount: pyx.int

    histBins:    list
    histWeights: list
    histN:       list
    histCount:   pyx.int

    def __init__(self, codes, operands, stackSize=100, memorySize=100) -> None:
        self.codes    = codes
        self.operands = operands
        self.stack    = np.zeros(stackSize)
        self.memory   = np.zeros(memorySize)
        self.stackSize  = stackSize
        self.stackCount = 0
        self.memoryCount = 0

        self._codes    = self.codes
        self._operands = self.operands
        self._stack    = self.stack
        self._memory   = self.memory

        self.histBins    = []
        self.histWeights = []
        self.histN       = []
        self.histCount   = 0

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def pushStack(self, value: pyx.double):
        self._stack[self.stackCount] = value
        self.stackCount += 1

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def popStack(self) -> pyx.double:
        assert self.stackCount > 0
        self.stackCount -= 1
        return self._stack[self.stackCount]

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _reset(self) -> pyx.void:
        self.stackCount  = 0
        self.memoryCount = 0

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _add(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 + x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _sub(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x2 - x1)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _mul(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 * x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _div(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x2 / x1)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _mod(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x2 % x1)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _floordiv(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(pyx.cast(pyx.double, pyx.cast(pyx.long, x2) // pyx.cast(pyx.long, x1)))

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _pow(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x2 ** x1)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _max(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x1 > x2:
            self.pushStack(x1)
        else:
            self.pushStack(x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _min(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x1 < x2:
            self.pushStack(x1)
        else:
            self.pushStack(x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _neg(self) -> pyx.void:
        x1 = self.popStack()
        self.pushStack(-x1)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _abs(self) -> pyx.void:
        x1 = self.popStack()
        if x1 < 0:
            self.pushStack(-x1)
        else:
            self.pushStack(x1)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _gt(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x2 > x1:
            self.pushStack(1)
        else:
            self.pushStack(0)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _gte(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x2 >= x1:
            self.pushStack(1)
        else:
            self.pushStack(0)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _lt(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x2 < x1:
            self.pushStack(1)
        else:
            self.pushStack(0)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _lte(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        if x2 <= x1:
            self.pushStack(1)
        else:
            self.pushStack(0)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randInt(self) -> pyx.void:
        h = self.popStack()
        l = self.popStack()
        self.pushStack(pyx.cast(pyx.double, randint(l, h)))

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randNorm(self) -> pyx.void:
        stdev = self.popStack()
        mu    = self.popStack()
        self.pushStack(randnorm(mu, stdev))

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randExp(self) -> pyx.void:
        lam = self.popStack()
        self.pushStack(randexp(lam))

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randGamma(self) -> pyx.void:
        loc   = self.popStack()
        theta = self.popStack()
        k     = self.popStack()
        self.pushStack(randgamma(k, theta, loc))

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randPoisson(self) -> pyx.void:
        lam = self.popStack()
        self.pushStack(pyx.cast(pyx.double, randpoisson(lam)))

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randNegBinom(self) -> pyx.void:
        p = self.popStack()
        r = self.popStack()
        self.pushStack(pyx.cast(pyx.double, randnegbinom(r, p)))

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randPert(self) -> pyx.void:
        lam  = self.popStack()
        high = self.popStack()
        mode = self.popStack()
        low  = self.popStack()
        self.pushStack(randpert(low, mode, high, lam))

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randHist(self, histIdx: pyx.int) -> pyx.void:
        bins:    pyx.double[:] = self.histBins[histIdx]
        weights: pyx.double[:] = self.histWeights[histIdx]
        n: pyx.int             = self.histN[histIdx]

        # Draw a uniform sample in [0, total_weight)
        total: pyx.double = 0.0
        i: pyx.int
        for i in range(n):
            total += weights[i]

        u: pyx.double = rand() * total

        # Walk the weights to find the sampled bin
        cumulative: pyx.double = 0.0
        for i in range(n):
            cumulative += weights[i]
            if u < cumulative:
                self.pushStack(bins[i])
                return

        # Fallback: last bin
        self.pushStack(bins[n - 1])


    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def sample(self) -> pyx.double:
        """Execute the VM program once and return the result.

        Runs the compiled bytecode from start to finish, using any random
        sampling instructions to draw a single sample from the distribution.

        Returns:
            float: The result of executing the VM program once.
        """
        N: pyx.int = self._codes.shape[0]
        i: pyx.int = 0
        opCode: pyx.long
        histIdx: pyx.int

        while i < N:
            opCode = self._codes[i]

            if   opCode == _PASS:
                pass
            elif opCode == _PUSH:
                self.pushStack(self._operands[i])
            elif opCode == _DROP:
                self.popStack()
            elif opCode == _STORE:
                slot: pyx.int = pyx.cast(pyx.int, self._operands[i])
                self._memory[slot] = self.popStack()
            elif opCode == _LOAD:
                slot = pyx.cast(pyx.int, self._operands[i])
                self.pushStack(self._memory[slot])
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
            elif opCode == _MAX:
                self._max()
            elif opCode == _MIN:
                self._min()
            elif opCode == _GT:
                self._gt()
            elif opCode == _GTE:
                self._gte()
            elif opCode == _LT:
                self._lt()
            elif opCode == _LTE:
                self._lte()
            elif opCode == _RANDINT:
                self._randInt()
            elif opCode == _RANDNORM:
                self._randNorm()
            elif opCode == _RANDEXP:
                self._randExp()
            elif opCode == _RANDGAMMA:
                self._randGamma()
            elif opCode == _RANDPOISSON:
                self._randPoisson()
            elif opCode == _RANDNEGBINOM:
                self._randNegBinom()
            elif opCode == _RANDPERT:
                self._randPert()
            elif opCode == _RANDHIST:
                histIdx = pyx.cast(pyx.int, self._operands[i])
                self._randHist(histIdx)

            i += 1

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
            rv.add(x, 1)

        return rv

    def run(self):
        """Run the simulation with default parameters.

        Convenience method that calls compute() with default parameters.

        Returns:
            Digest: A t-digest containing the simulation results
        """
        return self.compute()

#==================================================================================================
