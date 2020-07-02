import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    # Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1
    out = torch.randn(weights.size())
    #thanks to this initialization, we have var(out) = std^2
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m):
    # Python trick that will look for the type of connection in the object "m" (convolution or fully connection)
    classname = m.__class__.__name__
    # if convolution layer
    if classname.find('Conv') != -1:
        # List of shape of weights of conv layer
        weight_shape = list(m.weight.data.size())
        # dim1 * dim2 * dim3
        fan_in = np.prod(weight_shape[1:4])
        # dim0 * (dim2 * dim3)
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        # weight bound
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        # generating some random weights of order inversely proportional to the size of the tensor of weights
        # fills the given 2-dimensional matrix with values drawn from a uniform distribution parameterized by low and high
        m.weight.data.uniform_(-w_bound, w_bound)
        # initializing all the bias with zeros
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        # List of shape of weights of fully connected layer
        weight_shape = list(m.weight.data.size())
        # dim1
        fan_in = weight_shape[1]
        # dim1
        fan_out = weight_shape[0]
        # weight bound
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.weight.data.uniform_(-w_bound, w_bound)
        # initializing all the bias with zeros
        m.bias.data.fill_(0)

class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        # Initilizing the weights of the model with random weights
        self.apply(weights_init)
        # Setting the standard deviation of the actor tensor of weights to 0.01
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        # Initializing the actor bias with zeros
        self.actor_linear.bias.data.fill_(0)
        # Setting the standard deviation of the critic tensor of weights to 1.0
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0)
        # Initializing the critic bias with zeros
        self.critic_linear.bias.data.fill_(0)
        # The learnable input-hidden bias of the kth layer. 
        self.lstm.bias_ih.data.fill_(0)
        # The learnable hidden-hidden bias of the kth layer. 
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        # flattening the last convolutional layer into this 1D vector x
        x = x.view(-1, 32 * 3 * 3)
        # the LSTM takes as input x and the old hidden a<t-1> & cell states c<t-1> and ouputs the new hidden & cell states
        # output a<t>, c<t>
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        # return value,probability distribution, (a<t>,c<t>)
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)