"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * node.inputs[0] ** (self.scalar - 1) * self.scalar,)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / rhs / rhs
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar, )
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        x0, x1 = -2, -1
        if not self.axes is None:
            x0, x1 = list(self.axes)
        perm = [*range(len(a.shape))]
        perm[x0], perm[x1] = perm[x1], perm[x0]
        perm = tuple(perm)
        return a.permute(perm)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad.transpose(self.axes), )
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # print('a', a.shape)
        #
        # for i in range(len(a.shape), len(out_grad.shape)):
        #     out_grad = out_grad.sum(i)
        # print('out_grad', out_grad.shape)
        return (out_grad.reshape(a.shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        d = len(out_grad.shape) - len(a.shape)
        axes = tuple([i for i, v in enumerate(list(a.shape)) if v == 1])
        return out_grad.sum(tuple(range(d))).sum(axes).reshape(a.shape),
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, (tuple, list)):
            tmp = a
            for axis in sorted(self.axes, reverse=True):
                tmp = array_api.summation(tmp, axis=axis)
            return tmp
        else:
            return array_api.summation(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None:
            shape = list(out_grad.shape)
            for x in sorted(self.axes) if isinstance(self.axes, tuple) else [self.axes]:
                shape.insert(x, 1)
            shape = tuple(shape)
        else:
            shape = (1,) * len(node.inputs[0].shape)
        return (out_grad.reshape(shape).broadcast_to(node.inputs[0].shape),)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        P, Q = out_grad.matmul(rhs.transpose()), lhs.transpose().matmul(out_grad)
        if len(P.shape) > len(lhs.shape):
            P = P.sum(tuple([*range(len(P.shape) - len(lhs.shape))])).reshape(lhs.shape)
        if len(Q.shape) > len(rhs.shape):
            Q = Q.sum(tuple([*range(len(Q.shape) - len(rhs.shape))])).reshape(rhs.shape)
        return P, Q
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad/a, )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return (out_grad * exp(a), )
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * (a > 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        data = node.realize_cached_data()
        data = (data > 0)
        return (out_grad * Tensor(data, device = out_grad.device), )
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        ZMAX = Z.max(axis=self.axes, keepdims=True).broadcast_to(Z.shape)
        ret = array_api.log(array_api.summation(array_api.exp(Z - ZMAX), axis=self.axes))
        ret += Z.max(axis=self.axes)
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].realize_cached_data()
        ZMAX = Z.max(axis=self.axes, keepdims=True)
        shape = ZMAX.shape
        dom = array_api.reshape(array_api.summation(array_api.exp(Z - ZMAX.broadcast_to(Z.shape)), axis=self.axes), shape)
        return (Tensor(array_api.exp(Z - ZMAX.broadcast_to(Z.shape)) / dom.broadcast_to(Z.shape), device=node.inputs[0].device) * out_grad.reshape(shape).broadcast_to(Z.shape),)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        tmp = exp(node.inputs[0] * 2)
        return ((tmp + 2 + tmp ** -1) ** -1 * 4 * out_grad,)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        for i in range(1, len(args)):
            assert args[i].shape == args[i - 1].shape
        perm = tuple([(i + 1) % (self.axis + 1) if i <= self.axis else i for i in range(len(args[0].shape) + 1)])
        return array_api.array([x.numpy() for x in args], device=args[0].device).permute(perm)
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        perm = tuple([self.axis] + [i if i < self.axis else i + 1 for i in range(len(a.shape) - 1)])
        a = a.permute(perm)
        shard_shape = a.shape[1:]
        stride = 1
        for x in shard_shape:
            stride *= x
        a = a.compact().reshape((a.shape[0], stride)).compact()
        return tuple([NDArray.make(shape=shard_shape, device=a.device, handle=a._handle, offset=i*stride) for i in range(a.shape[0])])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (stack(out_grad, self.axis),)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Tensor(out_grad.realize_cached_data().flip(self.axes), device=out_grad.device)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        index = []
        for i in range(a.ndim):
            if i in self.axes:
                new_shape[i] *= self.dilation + 1
                index.append(slice(0, new_shape[i], self.dilation + 1))
            else:
                index.append(slice(0, new_shape[i], 1))
        ret = NDArray.make(new_shape, device=a.device)
        ret.fill(0)
        ret[tuple(index)] = a
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(a.shape)
        res_index, index = [], []
        for i in range(a.ndim):
            if i in self.axes:
                index.append(slice(0, new_shape[i], self.dilation + 1))
                new_shape[i] //= self.dilation + 1
            else:
                index.append(slice(0, new_shape[i], 1))
            res_index.append(slice(0, new_shape[i], 1))
        ret = NDArray.make(new_shape, device=a.device)
        ret[tuple(res_index)] = a[tuple(index)]
        return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        S, P = self.stride, self.padding
        if P > 0: A = A.pad(((0, 0), (P, P), (P, P), (0, 0)))

        N, H_in, W_in, C_in = A.shape
        K, _, _, C_out = B.shape
        H_out, W_out = (H_in - K) // S + 1, (W_in - K) // S + 1
        
        N_s, H_s, W_s, C_s = A.strides
        inner_dim = K * K * C_in
        A_s = A.as_strided(shape=(N, H_out, W_out, K, K, C_in), strides = (N_s, H_s*S, W_s*S, H_s, W_s, C_s))
        A_s = A_s.compact().reshape((N*H_out*W_out, inner_dim))
        out = A_s @ B.compact().reshape((inner_dim, C_out))
        return out.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        S, P = self.stride, self.padding
        A, B = node.inputs
        K, _, _, C_out = B.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, axes=(1,2), dilation=self.stride-1)

        # A.grad
        
        # print(f'{A.shape=}')
        # print(f'{out_grad.shape=}')
        B_transpose_flip = transpose(flip(B, (0, 1)), (2, 3))
        # print(f'{B_transpose_flip.shape=}')
        new_padding = K - P - 1
        A_grad = conv(out_grad, B_transpose_flip, padding=new_padding)
        # print(f'{A_grad.shape=}')

        A_transpose = transpose(transpose(A, (1, 2)), (0, 3))
        # print(f'{A_transpose.shape=}')
        out_grad_transpose = transpose(out_grad, (0, 2))
        # print(f'{out_grad_transpose.shape=}')
        B_grad = conv(A_transpose, out_grad_transpose, padding=P)
        B_grad = transpose(B_grad, (0, 2))
        # print(f'{B_grad.shape=}')
        return (A_grad, B_grad)
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



