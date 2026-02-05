import re
import pathlib

# BEGIN
OP_PASS = 0
OP_PUSH = 1
OP_DROP = 2
OP_STORE = 3
OP_LOAD = 4
OP_NEG = 10
OP_ABS = 11
OP_ADD = 20
OP_MUL = 21
OP_POW = 22
OP_DIV = 23
OP_SUB = 24
OP_MOD = 25
OP_FLOORDIV = 26
OP_LT = 27
OP_LE = 28
OP_MAX = 29
OP_MIN = 29
OP_BINOPMAX = 50
OP_SUM_START = 51
OP_SUM_END = 52
OP_RANDINT = 100
OP_RANDNORM = 101
OP_RAND_QUANTILES = 102
OP_ARRAY_SUM = 103
OP_RAND_HIST = 104

# END

if __name__ == "__main__":
    ths = pathlib.Path(__file__).absolute()
    pth = (pathlib.Path(__file__).absolute() / '..' / 'core.py').resolve()

    thisFile = open(ths, 'r').readlines()

    with open(pth, 'r') as fin:

        with open('opcodes.py', 'w') as fout:
            for e, lo in enumerate(thisFile):
                fout.write(lo)
                if lo.startswith("# BEGIN"):
                    break

            for l in fin:
                if mtch := re.match("^(_[A-Z_]+)\s+=\s?pyx.declare\(pyx.int,\s?([0-9]+)\)$", l):
                    code = mtch.groups()[0]
                    numb = int(mtch.groups()[1])
                    out = f'OP{code} = {numb}\n'
                    fout.write(out)
                    print(out)
            e += 1
            while e < len(thisFile):
                fout.write(thisFile[e])
                e += 1
