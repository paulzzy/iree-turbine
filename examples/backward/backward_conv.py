import torch
import unittest
import logging


def conv(x, w, b):
    return torch.convolution(
        x,
        w,
        b,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
    )


def run_backward(x, w, b, func):
    x0 = x.clone().requires_grad_(True)
    w0 = w.clone().requires_grad_(True)
    b0 = b.clone().requires_grad_(True)
    # compile the forward convolution
    y = func(x0, w0, b0)
    # run backward to compile backward graph
    y.sum().backward()
    return x0.grad, w0.grad, b0.grad


class BackwardConv(unittest.TestCase):
    def testConvBackward(self):
        # generate random inputs
        x = torch.randn((2, 3, 10, 10))
        w = torch.randn((1, 3, 2, 2))
        b = torch.randn((1))
        turbine_conv = torch.compile(conv, backend="turbine_cpu")
        # clone for grad computation with compiled func
        x_grad0, w_grad0, b_grad0 = run_backward(x, w, b, turbine_conv)
        x_grad1, w_grad1, b_grad1 = run_backward(x, w, b, conv)
        # compare grads
        assert torch.allclose(x_grad0, x_grad1)
        assert torch.allclose(w_grad0, w_grad1)
        assert torch.allclose(b_grad0, b_grad1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
