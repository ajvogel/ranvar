from .digest import TDigest
import cython as pyx
import numpy as np

#from cython.cimports.digest import TDigest
from cython.cimports.casino.random import randint

#from .random import randint
# Virtual Machine


@pyx.cclass
class Engine():

    __PASS__:pyx.int    = 0
    __PUSH__:pyx.int    = 1
    __ADD__:pyx.int     = 2
    __SUB__:pyx.int     = 3
    __MUL__:pyx.int     = 4
    __DIV__:pyx.int     = 5
    __POW__:pyx.int     = 6
    __RANDINT__:pyx.int = 7



    _codes: pyx.long[:]
    _operands: pyx.double[:]
    _stack: pyx.double[:]

    codes: np.ndarray
    operands: np.ndarray
    stack: np.ndarray
    stackCount: pyx.int
    def __init__(self, codes, operands) -> None:
        self.codes    = codes
        self.operands = operands
        self.stack    = np.zeros(100)
        self.stackCount = 0

    # Init the memory view.fdd
        self._codes = self.codes
        self._operands = self.operands
        self._stack   = self.stack

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
        self.pushStack(x1 - x2)

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
        self.pushStack(x1 / x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _mod(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 % x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _pow(self) -> pyx.void:
        x1 = self.popStack()
        x2 = self.popStack()
        self.pushStack(x1 ** x2)

    @pyx.cfunc
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _randInt(self) -> pyx.void:
        h = self.popStack()
        l = self.popStack()

        # x: pyx.double = c_round((h - l)*_rand() + l)

        self.pushStack(pyx.cast(pyx.double, randint(l,h)))

    @pyx.ccall
    @pyx.boundscheck(False)
    @pyx.wraparound(False)
    @pyx.initializedcheck(False)
    def _compute(self) -> pyx.float:
        # PASS:pyx.int    = 0
        # PUSH:pyx.int    = 1
        # ADD:pyx.int    = 2
        # MUL:pyx.int     = 3
        # POW:pyx.int     = 4
        # RANDINT:pyx.int = 5
        N:pyx.int = self._codes.shape[0]
        i:pyx.int = 0
        opCode: pyx.long

        while i < N:
            opCode = self._codes[i]

            if   opCode == self.__PASS__:
                pass
            elif opCode == self.__PUSH__:
                self.pushStack(self._operands[i])
            elif opCode == self.__ADD__:
                self._add()
            elif opCode == self.__SUB__:
                self._sub()
            elif opCode == self.__MUL__:
                self._mul()
            elif opCode == self.__DIV__:
                self._div()
            elif opCode == self.__POW__:
                self._pow()
            elif opCode == self.__RANDINT__:
                self._randInt()

            i += 1

        return self.popStack()

    def compute(self, samples:pyx.int=10_000, maxBins:pyx.int=32):
        rv = TDigest(maxBins=maxBins)
        i: pyx.int
        for i in range(samples):
            x:pyx.float = self._compute()
            rv._add(x, 1)

        return rv

    def sample(self):
        return self._compute()

    # def run(self):
    #     return self.compute()
