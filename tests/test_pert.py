import ranvar as rv
import numpy as np


# Theoretical mean of PERT(low, mode, high) = (low + 4*mode + high) / 6


def _compute(node, samples=50_000):
    return node.compute(samples=samples)


# --[ Opcode-level VM tests ]---------------------------------------------------

def test_pert_vm_bytecode():
    """PERT samples directly via VM bytecode fall within [low, high]."""
    low, mode, high = 1.0, 5.0, 10.0
    BYTECODE = [
        (rv.OP_PUSH, low),
        (rv.OP_PUSH, mode),
        (rv.OP_PUSH, high),
        (rv.OP_RAND_PERT, 0),
    ]
    opcodes  = np.array([op for op, _ in BYTECODE], dtype=np.double)
    operands = np.array([oe for _, oe in BYTECODE], dtype=np.double)

    vm = rv.VirtualMachine(opcodes, operands)
    d = vm.compute(samples=20_000)

    assert d.lower() >= low - 0.01
    assert d.upper() <= high + 0.01


def test_pert_vm_mean():
    """PERT VM samples have mean close to (low + 4*mode + high) / 6."""
    low, mode, high = 0.0, 50.0, 100.0
    expected_mean = (low + 4 * mode + high) / 6  # 50.0
    BYTECODE = [
        (rv.OP_PUSH, low),
        (rv.OP_PUSH, mode),
        (rv.OP_PUSH, high),
        (rv.OP_RAND_PERT, 0),
    ]
    opcodes  = np.array([op for op, _ in BYTECODE], dtype=np.double)
    operands = np.array([oe for _, oe in BYTECODE], dtype=np.double)

    vm = rv.VirtualMachine(opcodes, operands)
    d = vm.compute(samples=20_000)

    assert abs(d.mean() - expected_mean) < 2.0


# --[ Node / high-level API tests ]---------------------------------------------

def test_pert_node_mean_symmetric():
    """Symmetric PERT mean equals the mode."""
    low, mode, high = 0.0, 50.0, 100.0
    expected_mean = (low + 4 * mode + high) / 6  # 50.0

    d = _compute(rv.Pert(low=low, mode=mode, high=high))
    assert abs(d.mean() - expected_mean) < 2.0


def test_pert_node_mean_skewed_low():
    """Left-skewed PERT mean is close to (low + 4*mode + high) / 6."""
    low, mode, high = 0.0, 10.0, 100.0
    expected_mean = (low + 4 * mode + high) / 6  # ~26.67

    d = _compute(rv.Pert(low=low, mode=mode, high=high))
    assert abs(d.mean() - expected_mean) < 3.0


def test_pert_node_bounds():
    """PERT samples always lie within [low, high]."""
    low, mode, high = 5.0, 8.0, 15.0

    d = _compute(rv.Pert(low=low, mode=mode, high=high))
    assert d.lower() >= low - 0.01
    assert d.upper() <= high + 0.01


def test_pert_node_median_near_mode():
    """For a nearly-symmetric PERT the median is close to the mode."""
    low, mode, high = 0.0, 50.0, 100.0

    d = _compute(rv.Pert(low=low, mode=mode, high=high))
    assert abs(d.quantile(0.5) - mode) < 5.0


def test_pert_node_in_expression():
    """Pert node can participate in arithmetic expressions."""
    low, mode, high = 0.0, 10.0, 20.0
    expected_mean = (low + 4 * mode + high) / 6 + 100  # 10 + 100 = 110

    expr = rv.Pert(low=low, mode=mode, high=high) + rv.Constant(100)
    d = _compute(expr)
    assert abs(d.mean() - expected_mean) < 3.0


def test_pert_compile_produces_correct_opcodes():
    """Pert node compiles to the expected bytecode."""
    node = rv.Pert(low=1.0, mode=5.0, high=10.0)
    codes, operands = node.compile()

    assert rv.OP_RAND_PERT in list(codes)
    pert_idx = list(codes).index(rv.OP_RAND_PERT)
    push_operands = [operands[i] for i in range(pert_idx) if codes[i] == rv.OP_PUSH]
    assert 1.0  in push_operands
    assert 5.0  in push_operands
    assert 10.0 in push_operands
