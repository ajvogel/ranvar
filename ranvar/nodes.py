from typing import SupportsAbs
import numpy as np
from .digest import Digest
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
        """Initialize a random variable with child nodes.
        
        Args:
            *args: Variable number of child nodes (other RandomVariable instances or constants)
        """
        self.children = list(args)

    def __add__(self, other):
        """Addition operator (+).
        
        Args:
            other: Right operand (RandomVariable or numeric constant)
            
        Returns:
            Add: New addition node in the expression tree
        """
        return Add(self, other)

    def __radd__(self, other):
        """Reverse addition operator (supports constant + RandomVariable).
        
        Args:
            other: Left operand (typically a numeric constant)
            
        Returns:
            Add: New addition node in the expression tree
        """
        return Add(other, self)

    def __sub__(self, other):
        """Subtraction operator (-).
        
        Args:
            other: Right operand (RandomVariable or numeric constant)
            
        Returns:
            Sub: New subtraction node in the expression tree
        """
        return Sub(self, other)

    def __rsub__(self, other):
        """Reverse subtraction operator (supports constant - RandomVariable).
        
        Args:
            other: Left operand (typically a numeric constant)
            
        Returns:
            Sub: New subtraction node in the expression tree
        """
        return Sub(other, self)    

    def __mul__(self, other):
        """Multiplication operator (*).
        
        Args:
            other: Right operand (RandomVariable or numeric constant)
            
        Returns:
            Mul: New multiplication node in the expression tree
        """
        return Mul(self, other)

    def __rmul__(self, other):
        """Reverse multiplication operator (supports constant * RandomVariable).
        
        Args:
            other: Left operand (typically a numeric constant)
            
        Returns:
            Mul: New multiplication node in the expression tree
        """
        return Mul(other, self)    

    def __pow__(self, other):
        """Exponentiation operator (**).
        
        Args:
            other: Exponent (RandomVariable or numeric constant)
            
        Returns:
            Pow: New power node in the expression tree
        """
        return Pow(self, other)

    def __truediv__(self, other):
        """Division operator (/).
        
        Args:
            other: Divisor (RandomVariable or numeric constant)
            
        Returns:
            Div: New division node in the expression tree
        """
        return Div(self, other)

    def __mod__(self, other):
        """Modulo operator (%).
        
        Args:
            other: Divisor (RandomVariable or numeric constant)
            
        Returns:
            Mod: New modulo node in the expression tree
        """
        return Mod(self, other)

    def __floordiv__(self, other):
        """Floor division operator (//).
        
        Args:
            other: Divisor (RandomVariable or numeric constant)
            
        Returns:
            FloorDiv: New floor division node in the expression tree
        """
        return FloorDiv(self, other)

    def __matmul__(self, other):
        """Matrix multiplication operator (@) - used for summation.
        
        This operator is overloaded to represent summation operations
        in Monte Carlo contexts, typically for computing expected values
        or aggregating random variables.
        
        Args:
            other: Right operand (RandomVariable or numeric constant)
            
        Returns:
            Summation: New summation node in the expression tree
        """
        return Summation(self, other)

    def __rmatmul__(self, other):
        """Reverse matrix multiplication operator (supports constant @ RandomVariable).
        
        Args:
            other: Left operand (typically a numeric constant)
            
        Returns:
            Summation: New summation node in the expression tree
        """
        return Summation(other, self)

    def __divmod__(self, other):
        """Divmod operator - not implemented.
        
        Args:
            other: Right operand
        """
        pass

    def __rtruediv__(self, other):
        """Reverse division operator - not implemented.
        
        Args:
            other: Left operand
        """
        pass

    def __rfloordiv__(self, other):
        """Reverse floor division operator - not implemented.
        
        Args:
            other: Left operand
        """
        pass

    def __rdivmod__(self, other):
        """Reverse divmod operator - not implemented.
        
        Args:
            other: Left operand
        """
        pass

    def __lt__(self, other):
        """Less than operator (<).
        
        Args:
            other: Right operand (RandomVariable or numeric constant)
            
        Returns:
            LessThan: New less-than comparison node in the expression tree
        """
        return LessThan(self, other)

    def __le__(self, other):
        """Less than or equal operator (<=).
        
        Args:
            other: Right operand (RandomVariable or numeric constant)
            
        Returns:
            LessThanEqual: New less-than-or-equal comparison node in the expression tree
        """
        return LessThanEqual(self, other)

    def __gt__(self, other):
        """Greater than operator (>).
        
        Implemented by reversing the operands of less-than.
        
        Args:
            other: Right operand (RandomVariable or numeric constant)
            
        Returns:
            LessThan: New less-than comparison node with reversed operands
        """
        return LessThan(other, self)

    def __ge__(self, other):
        """Greater than or equal operator (>=).
        
        Implemented by reversing the operands of less-than-or-equal.
        
        Args:
            other: Right operand (RandomVariable or numeric constant)
            
        Returns:
            LessThanEqual: New less-than-or-equal comparison node with reversed operands
        """
        return LessThanEqual(other, self)

    def mean(self, samples=10000):
        return self.compute(samples=samples).mean()
        
    def printTree(self, level=0):
        """Print a visual representation of the expression tree.
        
        Recursively prints the tree structure with indentation to show
        the hierarchy of operations and operands.
        
        Args:
            level (int, optional): Current indentation level. Defaults to 0.
        """
        print(' '*level*4+self.__class__.__name__)
        for c in self.children:
            if hasattr(c, 'printTree'):
                c.printTree(level+1)
            else:
                print(' '*(level + 1)*4+str(c))

    def _compile(self, codes, operands):
        """Compile this node into VM bytecode (abstract method).
        
        This method should be overridden by subclasses to generate
        the appropriate bytecode instructions for their operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        pass

    def _compileOrPush(self, codes, operands, child):
        """Compile a child node or push a constant value.
        
        If the child is a RandomVariable, compile it recursively.
        Otherwise, treat it as a constant and generate a PUSH instruction.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
            child: Child node (RandomVariable) or constant value
        """
        if hasattr(child, '_compile'):
            child._compile(codes, operands)
        else:
            codes.append(OP_PUSH)
            operands.append(child)


    def _compileChildren(self, codes, operands):
        """Compile all child nodes in order.
        
        Helper method that compiles each child node, useful for operations
        that need to process multiple operands.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        for c in self.children:
            self._compileOrPush(codes, operands, c)

    def compile(self):
        """Compile the entire expression tree into VM bytecode.
        
        Traverses the expression tree and generates arrays of operation
        codes and operands that can be executed by the VirtualMachine.
        
        Returns:
            tuple: (codes, operands) as numpy arrays
                - codes: Array of operation codes
                - operands: Array of corresponding operands
        """
        codes    = []
        operands = []
        self._compile(codes, operands)

        codes = np.array(codes, dtype=np.double)
        operands = np.array(operands, dtype=np.double)
        return codes, operands

    def sample(self):
        """Generate a single sample from the random variable.
        
        Compiles the expression tree and executes it once using the
        VirtualMachine to produce a single random sample.
        
        Returns:
            float: A single sample from the distribution
        """
        codes, operands = self.compile()
        vm = VirtualMachine(codes, operands)
        return vm.sample()

    def compute(self, samples=10000, maxBins=32):
        """Perform Monte Carlo simulation to estimate the distribution.
        
        Compiles the expression tree and runs it multiple times to build
        a statistical approximation of the resulting distribution using
        a t-digest data structure.
        
        Args:
            samples (int, optional): Number of Monte Carlo samples. Defaults to 10000.
            maxBins (int, optional): Maximum bins for the t-digest. Defaults to 32.
            
        Returns:
            DigestVariable: A wrapper around the t-digest containing the
                           estimated distribution
        """
        # print('Compiling...')
        codes, operands = self.compile()
        vm = VirtualMachine(codes, operands)
        # print('Simulating...')
        digest =  vm.compute(samples=samples, maxBins=maxBins)
        return DigestVariable(digest)



#-----------------------------------------------------------------------------------------

class DigestVariable(RandomVariable):
    """
    Wrapper for Digest objects to enable use in algebraic operations.
    
    This class wraps a t-digest data structure in a RandomVariable interface,
    allowing statistical distributions computed from Monte Carlo simulations
    to be used as operands in further algebraic expressions.
    
    The DigestVariable provides access to the underlying t-digest's statistical
    methods (quantile, CDF, bounds) while also enabling the digest to be compiled
    into VM bytecode for sampling operations. When compiled, the digest is converted
    into a histogram representation suitable for random sampling.
    
    Attributes:
        _digest (Digest): The underlying t-digest data structure
        
    Example:
        >>> x = Normal(0, 1)
        >>> result = x.compute()  # Returns DigestVariable
        >>> median = result.quantile(0.5)
        >>> y = result + 10  # Can be used in further operations
    """
    def __init__(self, digest: Digest):
        """Initialize with a t-digest.
        
        Args:
            digest (Digest): The t-digest data structure to wrap
        """
        self._digest = digest

    def quantile(self, q):
        """Compute the quantile of the distribution.
        
        Args:
            q (float): Quantile probability between 0 and 1
            
        Returns:
            float: Estimated quantile value
        """
        return self._digest.quantile(q)

    def cdf(self, k):
        """Compute the cumulative distribution function.
        
        Args:
            k (float): Value at which to evaluate the CDF
            
        Returns:
            float: Estimated CDF value between 0 and 1
        """
        return self._digest.cdf(k)

    def lower(self):
        """Get the minimum value in the distribution.
        
        Returns:
            int: The smallest value in the digest
        """
        return self._digest.lower()

    def upper(self):
        """Get the maximum value in the distribution.
        
        Returns:
            int: The largest value in the digest
        """
        return self._digest.upper()

    def mean(self):
        return self._digest.mean()
    
    def _compile(self, codes, operands):
        """Compile the digest into VM bytecode for random sampling.
        
        Converts the t-digest into a histogram representation suitable for
        the virtual machine's random histogram sampling operation. The digest's
        centroids are transformed into (value, cumulative_probability) pairs
        that enable efficient inverse transform sampling.
        
        The compilation process:
        1. Extracts centroid values and weights from the digest
        2. Computes interval weights between adjacent centroids
        3. Builds cumulative probability distribution
        4. Pushes histogram data onto the VM stack in reverse order
        5. Emits OP_RAND_HIST instruction
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        # We convert and compile the digest node into a generic histogram for random
        # sampling.
        
        x = self._digest.getBins()
        w = self._digest.getWeights()
        n = self._digest.getActiveBinCount()

      

        # Trim possible zeros at the end of the arrays.
        x = x[:n]
        w = w[:n]

        # print([f'{xx:.1f}' for xx in x])
        # print([f'{ww:.1f}' for ww in w])          

        b = np.zeros(n - 1)


        
        c = np.zeros(n)

        for i in range(n - 1):
            b[i] = (w[i] + w[i+1]) / 2

        for i in range(n):
            for j in range(0, i - 1):
                c[i] = c[i] + b[j]

        c = c / b.sum()

        # print("Bins:")

        # print([f'{bb:.1f}' for bb in b])
        # print("C:")
        # print([f'{cc:.5f}' for cc in c])
        c2 = b.cumsum() / b.sum()        
        # print([f'{cc:.5f}' for cc in c2])        

        c[1:] = c2
        c[0]  = 0
        # print([f'{cc:.5f}' for cc in c])
        for i in range(n - 1, -1, -1):
            self._compileOrPush(codes, operands, c[i])
            self._compileOrPush(codes, operands, x[i])

        codes.append(OP_RAND_HIST)
        operands.append(n)
        
            
        
        
    
#-----------------------------------------------------------------------------------------

class Constant(RandomVariable):
    """Represents a constant value in the expression tree.
    
    A leaf node that wraps a numeric constant, allowing it to be used
    in algebraic expressions with other random variables.
    
    Example:
        >>> c = Constant(5)
        >>> x = Normal(0, 1) + c  # Adds constant 5 to normal distribution
    """
    def printTree(self,level=0):
        """Print the constant value with indentation.
        
        Args:
            level (int, optional): Indentation level. Defaults to 0.
        """
        print(' '*level*4+str(self.children[0]))

    def _compile(self, codes, operands):
        """Compile to PUSH operation for the constant value.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        codes.append(OP_PUSH)
        operands.append(self.children[0])

#-----------------------------------------------------------------------------------------

class RandInt(RandomVariable):
    """Generates random integers within a specified range.
    
    Takes two child nodes representing the lower and upper bounds
    and generates uniformly distributed random integers in that range.
    
    Example:
        >>> low = Constant(1)
        >>> high = Constant(10)
        >>> rand_int = RandInt(low, high)  # Random integer from 1 to 10
    """
    def _compile(self, codes, operands):
        """Compile to random integer generation operation.
        
        Compiles children (bounds) then emits RANDINT operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_RANDINT)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class LessThan(RandomVariable):
    """Implements less-than comparison (<) between two expressions.
    
    Returns 1.0 if the first operand is less than the second, 0.0 otherwise.
    Useful for creating indicator functions and conditional logic.
    
    Example:
        >>> x = Normal(0, 1)
        >>> indicator = x < 0  # Creates LessThan node
    """
    def _compile(self, codes, operands):
        """Compile to less-than comparison operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_LT)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class LessThanEqual(RandomVariable):
    """Implements less-than-or-equal comparison (<=) between two expressions.
    
    Returns 1.0 if the first operand is less than or equal to the second, 0.0 otherwise.
    Useful for creating indicator functions and conditional logic.
    
    Example:
        >>> x = Normal(0, 1)
        >>> indicator = x <= 0  # Creates LessThanEqual node
    """
    def _compile(self, codes, operands):
        """Compile to less-than-or-equal comparison operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_LE)
        operands.append(0)        

#-----------------------------------------------------------------------------------------

class Add(RandomVariable):
    """Implements addition between two expressions.
    
    Represents the sum of two random variables or expressions.
    Created automatically when using the + operator.
    
    Example:
        >>> x = Normal(0, 1)
        >>> y = Normal(5, 2)
        >>> sum_expr = x + y  # Creates Add node
    """
    def _compile(self, codes, operands):
        """Compile to addition operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_ADD)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Sub(RandomVariable):
    """Implements subtraction between two expressions.
    
    Represents the difference of two random variables or expressions.
    Created automatically when using the - operator.
    
    Example:
        >>> x = Normal(5, 1)
        >>> y = Normal(2, 1)
        >>> diff_expr = x - y  # Creates Sub node
    """
    def _compile(self, codes, operands):
        """Compile to subtraction operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_SUB)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Mul(RandomVariable):
    """Implements multiplication between two expressions.
    
    Represents the product of two random variables or expressions.
    Created automatically when using the * operator.
    
    Example:
        >>> x = Normal(1, 0.1)
        >>> y = Normal(2, 0.2)
        >>> product_expr = x * y  # Creates Mul node
    """
    def _compile(self, codes, operands):
        """Compile to multiplication operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_MUL)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Div(RandomVariable):
    """Implements division between two expressions.
    
    Represents the quotient of two random variables or expressions.
    Created automatically when using the / operator.
    
    Example:
        >>> x = Normal(10, 1)
        >>> y = Normal(2, 0.1)
        >>> ratio_expr = x / y  # Creates Div node
    """
    def _compile(self, codes, operands):
        """Compile to division operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_DIV)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class FloorDiv(RandomVariable):
    """Implements floor division between two expressions.
    
    Represents the floor of the quotient of two random variables or expressions.
    Created automatically when using the // operator.
    
    Example:
        >>> x = Normal(10, 1)
        >>> y = Normal(3, 0.1)
        >>> floor_div_expr = x // y  # Creates FloorDiv node
    """
    def _compile(self, codes, operands):
        """Compile to floor division operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_FLOORDIV)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Mod(RandomVariable):
    """Implements modulo operation between two expressions.
    
    Represents the remainder of division between two random variables or expressions.
    Created automatically when using the % operator.
    
    Example:
        >>> x = RandInt(1, 100)
        >>> remainder = x % 7  # Creates Mod node
    """
    def _compile(self, codes, operands):
        """Compile to modulo operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_MOD)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Pow(RandomVariable):
    """Implements exponentiation between two expressions.
    
    Represents one expression raised to the power of another.
    Created automatically when using the ** operator.
    
    Example:
        >>> x = Normal(2, 0.1)
        >>> power_expr = x ** 2  # Creates Pow node for x squared
    """
    def _compile(self, codes, operands):
        """Compile to exponentiation operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
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
    """Represents a normal (Gaussian) distribution.

    Generates random samples from a normal distribution with specified
    mean and standard deviation parameters.

    Attributes:
        mean: Mean of the distribution (default: 0)
        stdev: Standard deviation of the distribution (default: 1)

    Example:
        >>> x = Normal(0, 1)      # Standard normal
        >>> y = Normal(10, 2.5)   # Normal with mean=10, std=2.5
    """
    def __init__(self, mean=0, stdev=1):
        """Initialize normal distribution with mean and standard deviation.

        Args:
            mean (float, optional): Mean of the distribution. Defaults to 0.
            stdev (float, optional): Standard deviation. Defaults to 1.
        """
        self.mean = mean
        self.stdev = stdev

    def _compile(self, codes, operands):
        """Compile to normal distribution sampling operation.

        Pushes mean and standard deviation onto the stack, then emits
        the RANDNORM operation for sampling.

        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileOrPush(codes, operands, self.mean)
        self._compileOrPush(codes, operands, self.stdev)
        # self._compileChildren(codes, operands)
        codes.append(OP_RANDNORM)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class NegativeBinomial(RandomVariable):
    """Represents a negative binomial distribution.

    Generates random samples from a negative binomial distribution. Can be
    parameterized either by (n, p) or by (mean, dispersion).

    The negative binomial distribution models the number of failures before
    achieving n successes, where each trial has probability p of success.
    It is commonly used in count data modeling, especially when overdispersion
    is present (variance > mean).

    Parameterization options:
    1. Direct (n, p): n is number of successes, p is probability of success
    2. Mean-dispersion: mean and dispersion (alpha) where var = mean + alpha * mean^2

    Attributes:
        n: Number of successes (shape parameter)
        p: Probability of success in each trial

    Example:
        >>> x = NegativeBinomial(n=5, p=0.5)  # Direct parameterization
        >>> y = NegativeBinomial(mean=10, dispersion=0.5)  # Mean-dispersion form
    """
    def __init__(self, mean=None, dispersion=None, n=None, p=None):
        """Initialize negative binomial distribution.

        Either (n, p) or (mean, dispersion) must be provided.

        Args:
            mean (float, optional): Mean of the distribution.
            dispersion (float, optional): Dispersion parameter (alpha).
                Variance = mean + dispersion * mean^2
            n (float, optional): Number of successes (shape parameter).
            p (float, optional): Probability of success in each trial.
        """
        if mean is not None and dispersion is not None:
            # Convert mean-dispersion parameterization to n, p
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
        """Compile to negative binomial distribution sampling operation.

        Pushes n and p onto the stack, then emits the RAND_NEGBINOM
        operation for sampling.

        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileOrPush(codes, operands, self.n)
        self._compileOrPush(codes, operands, self.p)
        codes.append(OP_RAND_NEGBINOM)
        operands.append(0)

#-----------------------------------------------------------------------------------------

class Summation(RandomVariable):
    """
    Represents a mathematical summation (Σ) operation.
    
    Implements a loop-based summation where a term expression is evaluated
    multiple times and the results are accumulated. This is useful for modeling
    scenarios like portfolio values, aggregate risks, or any situation requiring
    the sum of multiple random outcomes.
    
    Attributes:
        nTerms: Number of iterations/terms to sum
        term: Expression to evaluate and sum for each iteration
        
    Example:
        >>> die = RandInt(1, 6)
        >>> sum_of_dice = Summation(10, die)  # Sum of 10 dice rolls
        >>> # Equivalent to: die + die + ... (10 times)
    """
    def __init__(self, nTerms=1, term=0):
        """Initialize summation with number of terms and expression.
        
        Args:
            nTerms (int or RandomVariable, optional): Number of terms to sum. Defaults to 1.
            term (RandomVariable or numeric, optional): Expression to sum. Defaults to 0.
        """
        self.nTerms = nTerms
        self.term   = term
    def _compile(self, codes, operands):
        """Compile to summation loop operations.
        
        Generates bytecode for a summation loop:
        1. Pushes number of terms
        2. Emits SUM_START to initialize loop
        3. Compiles the term expression (evaluated each iteration)
        4. Emits SUM_END to accumulate and loop
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """

        # Compile nTerms.
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
    """
    Samples from a distribution defined by equally spaced quantile values.
    
    Takes a series of quantile values and creates a distribution that can be
    sampled using linear interpolation between the quantiles. This is useful
    for modeling distributions based on expert knowledge or historical data
    points.
    
    The quantiles should represent equally spaced probability levels
    (e.g., 0%, 25%, 50%, 75%, 100% quantiles).
    
    Example:
        >>> # Define distribution by 5 quantiles
        >>> dist = Quantiles(10, 20, 30, 40, 50)
        >>> # Samples between these values with linear interpolation
    """
    def __init__(self, *args):
        """Initialize with quantile values.
        
        Args:
            *args: Variable number of quantile values in ascending order
        """
        self.children = list(reversed(args))

    def _compile(self, codes, operands):
        """Compile to quantile sampling operation.
        
        Pushes all quantile values onto the stack and emits the
        RAND_QUANTILES operation for interpolated sampling.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        codes.append(OP_RAND_QUANTILES)
        operands.append(len(self.children))


#-----------------------------------------------------------------------------------------
        
class ArraySum(RandomVariable):
    """
    Sums a contiguous subset of an array of random variables.
    
    Takes an array of expressions and sums only those within a specified
    range [start, end). This is useful for modeling partial sums, such as
    summing only profitable investments from a portfolio or selecting
    a subset of outcomes.
    
    Attributes:
        children: List of expressions (in reverse order for stack operations)
        start: Starting index of the range to sum
        end: Ending index of the range to sum (exclusive)
        
    Example:
        >>> portfolio = [Normal(100, 10) for _ in range(20)]  # 20 investments
        >>> partial_sum = ArraySum(portfolio, 5, 15)  # Sum investments 5-14
    """
    def __init__(self, array, start, end):
        """Initialize with array and range specification.
        
        Args:
            array (list): List of RandomVariable expressions
            start (int or RandomVariable): Starting index (inclusive)
            end (int or RandomVariable): Ending index (exclusive)
        """
        self.children = list(reversed(array))
        self.start = start
        self.end = end

    def _compile(self, codes, operands):
        """Compile to array summation operation.
        
        Pushes all array elements, then start and end indices,
        and emits the ARRAY_SUM operation.
        
        Args:
            codes (list): List to append operation codes to
            operands (list): List to append operands to
        """
        self._compileChildren(codes, operands)
        self._compileOrPush(codes, operands, self.end)
        self._compileOrPush(codes, operands, self.start)

        codes.append(OP_ARRAY_SUM)
        operands.append(len(self.children))
