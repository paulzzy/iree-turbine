import torch

from iree.turbine.ops._jinja_test_ops import test_add, LIBRARY
from iree.turbine.runtime.op_reg.base import ALL_CUSTOM_OP_REGS, fuse_custom_ops

test_add_op_instance = ALL_CUSTOM_OP_REGS["_turbine_jinja_test.test_add"]

op = fuse_custom_ops(
    test_add_op_instance, test_add_op_instance, {0: 0}, LIBRARY, "fused_add_add"
)

a0 = torch.ones([2, 1], device="cuda:0")
a1 = torch.ones([2, 1], device="cuda:0")
b1 = torch.ones([2, 1], device="cuda:0")

c0 = op(a0, a1, b1)
print(c0)
