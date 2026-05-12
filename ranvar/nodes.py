from typing import SupportsAbs
import numpy as np
from .cdigest import Digest
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
    
    Each operation creates a new node in an expression tree that can be compiled
    into bytecode for execution by the VirtualMachine class.
    
    Attributes:
        children (list): Child nodes in the expression tree
        
    Example:
        >>> x = Normal(0, 1)  # Normal distribution
        >>> y = Uniform(0, 10)  # Uniform distribution
        >>> z = x + y * 2  # Combined expression
        >>> result = z.compute()  # Monte Carlo simulation
    """
    def __init__(self, *args):
        """
        Args:
            *args: Variable number of child nodes (other RandomVariable instances or constants)
        """
        self.children = list(args)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)    

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)    

    def __pow__(self, other):
        return Pow(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __mod__(self, other):
        return Mod(self, other)

    def __floordiv__(self, other):
        return FloorDiv(self, other)

    def __matmul__(self, other):
        return Summation(self, other)

    def __rmatmul__(self, other):
        return Summation(other, self)

    def __divmod__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    def __rfloordiv__(self, other):
        pass

    def __rdivmod__(self, other):
        pass

    def __lt__(self, other):
        return LessThan(self, other)

    def __le__(self, other):
        return LessThanEqual(self, other)

    def __gt__(self, other):
        return LessThan(other, self)

    def __ge__(self, other):
        return LessThanEqual(other, self)

    def mean(self, samples=10000):
        return self.compute(samples=samples).mean()
        
    def printTree(self, level=0):
        print(' '*level*4+self.__class__.__name__)
        for c in self.children:
            if hasattr(c, 'printTree'):
                c.printTree(level+1)
            else:
                print(' '*(level + 1)*4+str(c))

    def _compile(self, codes, operands):
        pass

    def _compileOrPush(self, codes, operands, child):
        if hasattr(child, '_compile'):
            child._compile(codes, operands)
        else:
            codes.append(OP_PUSH)
            operands.append(child)


    def _compileChildren(self, codes, operands):
        for c in self.children:
            self._compileOrPush(codes, operands, c)

    def compile(self):
        codes    = []
        operands = []
        self._compile(codes, operands)

        codes = np.array(codes, dtype=np.double)
        operands = np.array(operands, dtype=np.double)
        return codes, operands

    def sample(self):
        codes, operands = self.compile()
        vm = VirtualMachine(codes, operands)
        return vm.sample()

    def compute(self, samples=10000, maxBins=32):
        codes, operands = self.compile()
        vm = VirtualMachine(codes, operands)
        digest =  vm.compute(samples=samples, maxBins=maxBins)
        return DigestVariable(digest)



#-----------------------------------------------------------------------------------------

class DigestVariable(RandomVariable):
    """
    Wrapper for Digest objects to enable use in algebraic operations.
    """
    def __init__(self, digest: Digest):
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
        return self._digest.lower()

    def upper(self):
        return self._digest.upper()

    def mean(self):
        return self._digest.mean()
    
    def _compile(self, codes, operands):
        x = self._digest.getBins()
        w = self._digest.getWeights()
        n = self._digest.getActiveBinCount()

        x = x[:n]
        w = w[:n]

        b = np.zeros(n - 1)
        c = np.zeros(n)

        for i in range(n - 1):
            b[i] = (w[i] + w[i+1]) / 2

        for i in range(n):
            for j in range(0, i - 1):
                c[i] = c[i] + b[j]

        c = c / b.sum()

        c2 = b.cumsum() / b.sum()        

        c[1:] = c2
        c[0]  = 0
        for i in range(n - 1, -1, -1):
            self._compileOrPush(codes, operands, c[i])
            self._compileOrPush(codes, operands, x[i])

        codes.append(OP_RAND_HIST)
        operands.append(n)
        

#-----------------------------------------------------------------------------------------

class Constant(RandomVariable):
    def printTree(self,level=0):
        print(' '*level*4+str(self.children[0]))

    def _compile(self, codes, operands):
        codes.append(OP_PUSH)
        operands.append(self.children[0])

#-----------------------------------------------------------------------------------------

class RandInt(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_RANDINT)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class LessThan(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_LT)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class LessThanEqual(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_LE)
        operands.append(0)        

#-----------------------------------------------------------------------------------------

class Add(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_ADD)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Sub(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_SUB)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Mul(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_MUL)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Div(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_DIV)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class FloorDiv(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_FLOORDIV)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Mod(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_MOD)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Pow(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_POW)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Max(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_MAX)
        operands.append(0)

class Min(RandomVariable):
    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_MIN)
        operands.append(0)          

#-----------------------------------------------------------------------------------------

class Normal(RandomVariable):
    def __init__(self, mean=0, stdev=1):
        self.mean = mean
        self.stdev = stdev

    def _compile(self, codes, operands):
        self._compileOrPush(codes, operands, self.mean)
        self._compileOrPush(codes, operands, self.stdev)
        codes.append(OP_RANDNORM)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class NegativeBinomial(RandomVariable):
    def __init__(self, mean=None, dispersion=None, n=None, p=None):
        if mean is not None and dispersion is not None:
            var = mean + dispersion * mean ** 2
            self.p = mean / var
            self.n = mean ** 2 / (var - mean)
        elif n is not None and p is not None:
            self.n = n
            self.p = p
        else:
            raise ValueError("Either (mean, dispersion) or (n, p) must be provided")

    def __repr__(self):
        return f'<NegativeBinomial(p={self.p}, n={self.n})>'

    def _compile(self, codes, operands):
        self._compileOrPush(codes, operands, self.n)
        self._compileOrPush(codes, operands, self.p)
        codes.append(OP_RAND_NEGBINOM)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Gamma(RandomVariable):
    def __init__(self, shape, scale, location=0):
        self.shape = shape
        self.scale = scale
        self.location = location

    def _compile(self, codes, operands):
        self._compileOrPush(codes, operands, self.shape)
        self._compileOrPush(codes, operands, self.scale)
        self._compileOrPush(codes, operands, self.location)
        codes.append(OP_RAND_GAMMA)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Pert(RandomVariable):
    def __init__(self, low, mode, high):
        self.low = low
        self.mode = mode
        self.high = high

    def _compile(self, codes, operands):
        self._compileOrPush(codes, operands, self.low)
        self._compileOrPush(codes, operands, self.mode)
        self._compileOrPush(codes, operands, self.high)
        codes.append(OP_RAND_PERT)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Summation(RandomVariable):
    def __init__(self, nTerms=1, term=0):
        self.nTerms = nTerms
        self.term   = term

    def _compile(self, codes, operands):
        if hasattr(self.nTerms, '_compile'):
            self.nTerms._compile(codes, operands)
        else:
            codes.append(OP_PUSH)
            operands.append(self.nTerms)

        codes.append(OP_SUM_START)
        operands.append(0)

        if hasattr(self.term, '_compile'):
            self.term._compile(codes, operands)
        else:
            codes.append(OP_PUSH)
            operands.append(self.term)

        codes.append(OP_SUM_END)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Quantiles(RandomVariable):
    def __init__(self, *args):
        self.children = list(reversed(args))

    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        codes.append(OP_RAND_QUANTILES)
        operands.append(len(self.children))


#-----------------------------------------------------------------------------------------
        
class ArraySum(RandomVariable):
    def __init__(self, array, start, end):
        self.children = list(reversed(array))
        self.start = start
        self.end = end

    def _compile(self, codes, operands):
        self._compileChildren(codes, operands)
        self._compileOrPush(codes, operands, self.end)
        self._compileOrPush(codes, operands, self.start)

        codes.append(OP_ARRAY_SUM)
        operands.append(len(self.children))
