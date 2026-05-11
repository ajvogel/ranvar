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
        self.stackCount   = 0
        self.iterCount    = 0
        self.pointerCount = 0
        self.counter      = 0

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
        self._pointers[self.pointerCount] = value
        self.pointerCount += 1

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popPointer(self) -> pyx.int:
        assert self.pointerCount > 0
        self.pointerCount -= 1
        return self._pointers[self.pointerCount]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def pushIterator(self, value: pyx.double) -> pyx.void:
        self._iterators[self.iterCount] = value
        self.iterCount += 1

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popIterator(self) -> pyx.double:
        assert self.iterCount > 0
        self.iterCount -= 1
        return self._iterators[self.iterCount]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def peekIterator(self) -> pyx.double:
        return self._iterators[self.iterCount - 1]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def peekPointer(self) -> pyx.int:
        return self._pointers[self.pointerCount - 1]

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def pushStack(self, value: pyx.double):
        self._stack[self.stackCount] = value
        self.stackCount += 1

    @pyx.cfunc
    @pyx.initializedcheck(False)
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    def popStack(self) -> pyx.float:
        # assert self.stackCount > 0
        self.stackCount -= 1
        return self._stack[self.stackCount]

    @pyx.cfunc
    def _dropStack(self, cnt:pyx.int = 1):
        i: pyx.int
        for i in range(cnt):
            self.popStack()

    @pyx.cfunc
    def _store(self, varNumber: pyx.double) -> pyx.void:
        idx: pyx.int = pyx.cast(pyx.int, varNumber)
        varValue = self.popStack()
        self._variables[idx] = varValue

    @pyx.cfunc
    def _load(self, varNumber: pyx.double) -> pyx.void:
        idx: pyx.int = pyx.cast(pyx.int, varNumber)
        self.pushStack(self._variables[idx])

    @pyx.cfunc
    def _sumStart(self, loopNumber: pyx.double) -> pyx.void:
        idx: pyx.int = pyx.cast(pyx.int, loopNumber)
        nTerms = self.popStack()
        self.pushStack(0)
        self.pushIterator(nTerms)
        self.pushPointer(self.counter)

    @pyx.cfunc
    def _sumEnd(self, loopNumber: pyx.double) -> pyx.void:
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
        h = self.popStack()
        l = self.popStack()

        self.pushStack(randint(l, h))

    @pyx.cfunc
    def _randNorm(self) -> pyx.void:
        std: pyx.double = self.popStack()
        mu: pyx.double    = self.popStack()

        self.pushStack(randnorm(mu, std))

    @pyx.cfunc
    def _randNegBinom(self) -> pyx.void:
        p: pyx.double = self.popStack()
        n: pyx.double = self.popStack()

        self.pushStack(randnegbinom(n, p))

    @pyx.cfunc
    def _randGamma(self) -> pyx.void:
        location: pyx.double = self.popStack()
        scale: pyx.double = self.popStack()
        shape: pyx.double = self.popStack()

        self.pushStack(randgamma(shape, scale) + location)

    @pyx.cfunc
    def _randPert(self) -> pyx.void:
        high: pyx.double = self.popStack()
        mode: pyx.double = self.popStack()
        low:  pyx.double = self.popStack()

        self.pushStack(randpert(low, mode, high))

    @pyx.cfunc
    def _arraySum(self, nArray: pyx.double) -> pyx.void:
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
        dY: pyx.double = 1. / (nBins - 1)
        y_: pyx.double = rand()

        i: pyx.double = c_floor(y_ / dY)

        self._dropStack(pyx.cast(pyx.int, i))

        xi:pyx.double = self.popStack()
        xi_n:pyx.double = self.popStack()

        yi:pyx.double   = i*dY
        yi_n:pyx.double = yi + dY

        self._dropStack(pyx.cast(pyx.int, nBins - i - 2))

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

        p = rand()

        xi = self.popStack()
        ci = self.popStack()

        x_ = 0

        for i_n in range(1, nB):
            xi_n = self.popStack()
            ci_n = self.popStack()

            if ci <= p < ci_n:
                m  = (xi_n - xi) / (ci_n - ci)
                x_ = xi + m*(p - ci)

            elif (i_n == nBins-1) and (p == 1):
                x_ = xi_n

            xi = xi_n
            ci = ci_n


        self.pushStack(x_)





    def printState(self):
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

        N:pyx.int = self._codes.shape[0]
        opCode: pyx.double
        operand: pyx.double

        while self.counter < N:
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
            elif opCode == _RAND_GAMMA:
                self._randGamma()
            elif opCode == _RAND_PERT:
                self._randPert()

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
