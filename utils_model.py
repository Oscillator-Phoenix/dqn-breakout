import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()

        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(512, action_dim)

        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        return self.__fc2(x)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")


class DuelingDQN(nn.Module):

    def __init__(self, action_dim, device):
        super(DuelingDQN, self).__init__()

        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        self.__fc1_v = nn.Linear(64*7*7, 256)
        self.__fc1_a = nn.Linear(64*7*7, 256)

        self.__fc2_v = nn.Linear(256, 1)
        self.__fc2_a = nn.Linear(256, action_dim)

        self.__action_dim = action_dim
        self.__device = device

        print("init DuelingDQN")

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))

        v = F.relu(self.__fc1_v(x.view(x.size(0), -1)))
        v = self.__fc2_v(v)

        a = F.relu(self.__fc1_a(x.view(x.size(0), -1)))
        a = self.__fc2_a(a)
        mean_a = torch.mean(a, dim=1, keepdim=True)
        a -= mean_a

        q = v + a

        # print("\n x shape", x.size())
        # print("\n v shape", v.size())
        # print("\n a shape", a.size())
        # print("\n q shape", q.size())

        return q

    @ staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
