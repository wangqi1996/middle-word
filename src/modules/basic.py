import torch
import torch.nn as nn
from src.utils import init

class Bottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''\
    """
    一个类father继承Bottle，然后就会在执行father的时候先调用Bottle的forward方法，来改变维度，
    后Bottle主动调用father的forward方法， 然后在执行father的forward的后续方法把维度修改回去
    """
    # input: [batch_size, q_len, m_len]
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)

class BatchBottle(nn.Module):
    ''' Perform the reshape routine before and after an operation '''

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0]*size[1]))
        return out.view(-1, size[0], size[1])

class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__()

        self.linear = nn.Linear(*args, **kwargs)
        self.reset_parameters()

    def forward(self, x):
        return self.linear(x)

    def reset_parameters(self):
        init.default_init(self.linear.weight)

class BottleLinear(Bottle, Linear):
    ''' Perform the reshape routine before and after a linear projection '''
    pass

class BottleSoftmax(Bottle, nn.Softmax):
    ''' Perform the reshape routine before and after a softmax operation'''
    # 只会调用Bottle的forward方法
    pass
