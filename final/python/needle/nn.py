"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias:
            return X @ self.weight + self.bias.broadcast_to((X.shape[0], self.out_features))
        else:
            return X @ self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        bsz, *other_dims = X.shape
        return ops.reshape(X, (bsz, np.prod(other_dims)))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(ops.add_scalar(ops.exp(-x), 1), scalar=-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m, k = logits.shape
        y_one_hot = init.one_hot(k, y, device=logits.device)
        ans = ops.logsumexp(logits, axes=(1,)) - (logits * y_one_hot).sum((1,))
        return ans.sum() / m
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, k = x.shape
        if self.training:
            mean = x.sum(axes=(0, )) / n
            self.running_mean.data = ((1 - self.momentum) * self.running_mean + self.momentum * mean).data
            mean = mean.reshape((1, k)).broadcast_to(x.shape)
            var = ((x - mean) ** 2).sum(axes=(0,)) / n
            self.running_var.data = ((1 - self.momentum) * self.running_var + self.momentum * var).data
            var = var.reshape((1, k)).broadcast_to(x.shape)
        else:
            mean = self.running_mean.reshape((1, k)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, k)).broadcast_to(x.shape)
        weight = self.weight.reshape((1, k)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, k)).broadcast_to(x.shape)
        return weight * (x - mean) / ((var + self.eps) ** 0.5) + bias

        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, k = x.shape
        mean = x.sum(axes=(1, )) / k
        mean = mean.reshape((n, 1)).broadcast_to(x.shape)
        var = ((x - mean) ** 2).sum(axes=(1, )) / k
        var = var.reshape((n, 1)).broadcast_to(x.shape)
        weight = self.weight.reshape((1, k)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, k)).broadcast_to(x.shape)
        return weight * (x - mean) / ((var + self.eps) ** 0.5) + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        mask = init.randb(*x.shape, p=1 - self.p)
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        K, I, O = kernel_size, in_channels, out_channels
        self.weight = Parameter(init.kaiming_uniform(K*K*I, K*K*O, (K, K, I, O), device=device, dtype=dtype)) 

        
        alpha = 1.0 / (K*K*I) ** 0.5
        self.bias = Parameter(init.rand(O, low=-alpha, high=alpha, device=device, dtype=dtype)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(_x, self.weight, stride = self.stride, padding = self.kernel_size // 2)
        out = out.transpose((2, 3)).transpose((1, 2))
        
        if self.bias:
            bias = ops.broadcast_to(ops.reshape(self.bias, shape=(1,self.out_channels,1,1)), out.shape)
            out += bias
        return out
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device=device
        self.dtype=dtype

        I, H= input_size, hidden_size
        alpha = (1./H) ** 0.5
        self.W_ih = Parameter(init.rand(I, H, low=-alpha, high=alpha, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(H, H, low=-alpha, high=alpha, device=device, dtype=dtype, requires_grad=True))
        self.bias_ih = Parameter(init.rand(H, low=-alpha, high=alpha, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(H, low=-alpha, high=alpha, device=device, dtype=dtype, requires_grad=True)) if bias else None

        self.activation = ops.tanh if nonlinearity == 'tanh' else ops.relu
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, _ = X.shape
        H = self.W_hh.shape[0]
        out = X @ self.W_ih
        if h is None:
            h = init.zeros(bs, H, device=self.device, dtype=self.dtype, requires_grad=True)
        out += h @ self.W_hh
        if self.bias_ih:
            out += (self.bias_ih + self.bias_hh).reshape((1, H)).broadcast_to(out.shape)

        return self.activation(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        for _ in range(1, num_layers):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        T, bs, I = X.shape
        L, H = self.num_layers, self.hidden_size

        if h0 == None:
            h = [None] * L
        else:
            h0 = ops.split(h0, 0)
            h = [ops.tuple_get_item(h0, i).reshape((bs, H)) for i in range(L)]

        x = ops.split(X, 0)
        Y = []
        for t in range(T):
            y = ops.tuple_get_item(x, t).reshape((bs, I))
            for l in range(L):
                # print(f'{y.shape=}', f'{h[l].shape=}')
                y = self.rnn_cells[l](y, h[l])
                h[l] = y
            Y.append(y)
    
        return ops.stack(Y, 0), ops.stack(h, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        I, H= input_size, hidden_size
        alpha = (1./H) ** 0.5
        self.W_ih = Parameter(init.rand(I, 4*H, low=-alpha, high=alpha, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(H, 4*H, low=-alpha, high=alpha, device=device, dtype=dtype, requires_grad=True))
        self.bias_ih = Parameter(init.rand(4*H, low=-alpha, high=alpha, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(4*H, low=-alpha, high=alpha, device=device, dtype=dtype, requires_grad=True)) if bias else None
        
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        
        bs, H = X.shape[0], self.W_hh.shape[0]
        out = X @ self.W_ih

        if h is None:
            h0 = init.zeros(bs, H, device=self.device, dtype=self.dtype, requires_grad=False)
            c = init.zeros(bs, H, device=self.device, dtype=self.dtype, requires_grad=False)
        else:
            h0, c = h
            if h0 is None:
                h0 = init.zeros(bs, H, device=self.device, dtype=self.dtype, requires_grad=False)
            if c is None:
                c = init.zeros(bs, H, device=self.device, dtype=self.dtype, requires_grad=False)

        out += h0 @ self.W_hh

        if self.bias_ih:
            out += (self.bias_ih + self.bias_hh).reshape((1, 4*H)).broadcast_to(out.shape)
        
        out = ops.split(out.reshape((bs, 4, H)), axis=1)
        i, f, g, o = self.sigmoid(out[0]), self.sigmoid(out[1]), self.tanh(out[2]), self.sigmoid(out[3])

        c_out = f*c + i*g
        h_out = o * self.tanh(c_out)
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, 
                    bias=bias, device=device, dtype=dtype) for i in range(num_layers)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        T, bs, I = X.shape
        L, H = self.num_layers, self.hidden_size

        if h is None:
            h = [None] * L
            c = [None] * L
        else:
            h0, c0 = h
            h0 = ops.split(h0, 0)
            c0 = ops.split(c0, 0)
            h = [ops.tuple_get_item(h0, i).reshape((bs, H)) for i in range(L)]
            c = [ops.tuple_get_item(c0, i).reshape((bs, H)) for i in range(L)]
        
        x = ops.split(X, 0)
        Y = []
        for t in range(T):
            y = ops.tuple_get_item(x, t).reshape((bs, I))
            for l in range(L):
                hh, cc = self.lstm_cells[l](y, (h[l], c[l]))
                y, h[l], c[l] = hh, hh, cc
            Y.append(y)
        return ops.stack(Y, 0), (ops.stack(h, 0), ops.stack(c, 0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, 
                                mean=0, std=1, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_split = ops.split(x, 0)
        y = []
        for i in range(seq_len):
            x_b = ops.tuple_get_item(x_split, i)
            x_one_hot = init.one_hot(self.num_embeddings, x_b, self.device, self.dtype, requires_grad=False)
            # print(f"{x_one_hot.shape=}")
            y.append(x_one_hot @ self.weight)
        return ops.stack(y, 0)
        ### END YOUR SOLUTION
