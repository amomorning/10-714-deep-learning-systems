import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def ConvBN(A, B, K, S, device=None, dtype="float32"):
    return nn.Sequential(
        nn.Conv(A, B, K, stride = S, bias = True, device = device), 
        nn.BatchNorm2d(dim = B, device = device),
        nn.ReLU()
    )

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.modules = nn.Sequential(
            ConvBN(3, 16, 7, 4, device=device, dtype=dtype),
            ConvBN(16, 32, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBN(32, 32, 3, 1, device=device, dtype=dtype),
                    ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
                )
            ),
            ConvBN(32, 64, 3, 2, device=device, dtype=dtype),
            ConvBN(64, 128, 3, 2, device=device, dtype=dtype),
            nn.Residual(
                nn.Sequential(
                    ConvBN(128, 128, 3, 1, device=device, dtype=dtype),
                    ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
                )
            ),
            nn.Flatten(),
            nn.Linear(128, 128, bias=True, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 10, bias=True, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.modules(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size, device, dtype)
        if seq_model == 'rnn':
            self.sequence = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.sequence = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            raise
        
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x = self.embedding(x) #(seq_len, bs, embedding_size)
        out, h = self.sequence(x, h) #(seq_len, bs, hidden_size)
        out = out.reshape((seq_len*bs, self.hidden_size))
        out = self.linear(out)
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    # print(dataset[1][0].shape)
