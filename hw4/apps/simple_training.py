import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    tot_acc, tot_loss = 0, 0
    for X, y in dataloader:
        logits = model(X)
        loss = loss_fn(logits, y)
        tot_loss += loss.detach().numpy() * y.shape[0]
        y_hat = np.argmax(logits.detach().numpy(), axis=1)
        tot_acc += np.sum(y_hat == y.numpy())
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    n = len(dataloader.dataset)
    return tot_acc / n, tot_loss / n
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        start_time = time.time()
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model,
                                                  loss_fn=loss_fn(), opt=opt)
        end_time = time.time()
        print(f"train {epoch}: {avg_acc=}, {avg_loss=}, time cost:{end_time - start_time}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model,
                                              loss_fn=loss_fn(), opt=None)
    print(f"evaluate: {avg_acc = }, {avg_loss = }")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    
    tot_acc, tot_loss = 0, 0
    nbatch, batch_size = data.shape
    h = None
    
    tot_samples, tot_batches = 0, 0
    for i in range(0, nbatch, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        out, h = model(X, h)
        if isinstance(h, tuple):
            h = tuple([h0.data for h0 in list(h)])
        else:
            h = h.data
        loss = loss_fn(out, y)
        tot_loss += loss.detach().numpy()
        y_hat = np.argmax(out.detach().numpy(), axis=1)
        tot_acc += np.sum(y_hat == y.numpy()) 
        tot_samples += y.shape[0]
        tot_batches += 1
        if opt:
            opt.reset_grad()
            loss.backward()
            if clip:
                opt.clip_grad_norm(clip)
            opt.step()
    return tot_acc / tot_samples, tot_loss / tot_batches
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        start_time = time.time()
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, 
                                                  loss_fn=loss_fn(), opt=opt, clip=clip,
                                                   device=device, dtype=dtype)
        end_time = time.time()
        print(f"train {epoch}: {avg_acc=}, {avg_loss=}, time cost:{end_time - start_time}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn=loss_fn(), 
                                        device=device, dtype=dtype)
    print(f"evaluate: {avg_acc = }, {avg_loss = }")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
