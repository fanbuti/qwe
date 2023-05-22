import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import copy


class HW(nn.Module):
    def __init__(self, beta):
        super(HW, self).__init__()
        '''
        VO
        OV
        VS
        SV
        OS
        SO

        SVO=VO
        SOV=OV
        VSO=VS+SO+3
        VOS=VO+OS+3
        OVS=OV+VS+3
        OSV=OS+SV+3
        '''
        test = torch.tensor(
            [[random.uniform(40, 60)], [random.uniform(40, 60)], [random.uniform(40, 60)], [random.uniform(40, 60)],
             [random.uniform(40, 60)], [random.uniform(40, 60)]])
        # test = torch.tensor([[20.0], [10.0], [8.0], [5.0], [1.0], [2.0]])
        # test = torch.tensor([])
        # if beta == 0.05:
        #     test = torch.tensor([[120.0], [60.0], [55.0], [15.0], [45.0], [70.0]])
        # if beta == 0.10:
        #     test = torch.tensor([[70.0], [30.0], [25.0], [5.0], [20.0], [30.0]])
        # if beta == 0.50:
        #     test = torch.tensor([[20.0], [15.0], [12.0], [10.0], [5.0], [7.5]])
        self.fc = nn.Linear(1, 6)
        self.fc.weight = nn.Parameter(test)
        self.coefficient = torch.tensor([[
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0]
        ]], dtype=torch.float)
        self.offset = torch.tensor([0, 0, 3, 3, 3, 3])
        self.beta = beta

    def forward(self, x):
        x = x.reshape(1, 1)
        x = self.fc(x)
        x = x.reshape(1, 1, 6)
        E = torch.bmm(x, self.coefficient)
        E = E.reshape(6)
        # print(E.shape)
        E = E + self.offset
        # print(E.shape)
        E = -E * self.beta
        # print(E.shape)
        p = F.softmax(E, dim=0)
        # print(p.shape)
        return p


train = torch.tensor([1.0], dtype=torch.float)
label = torch.tensor([0.42, 0.45, 0.09, 0.02, 0.01, 0.01], dtype=torch.float)


count = 5000

betas = [0.05, 0.10, 0.50]

MSE_Loss = []
costs = []


for beta in betas:
    loss_result = []
    result = []
    cost_result = []
    for T in range(20):
        model = HW(beta)
        loss_function = nn.MSELoss(reduction="mean")
        # optimizer = optim.SGD(model.parameters(), 0.000001)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        p = []
        for epoch in range(count):
            if epoch == 3000:
                lr = 1e-4
                for param_group in optimizer.param_groups:
                    # 更改全部的学习率
                    param_group['lr'] = lr

            input = train
            output = model(input)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            p.append(loss.item())
            # if (epoch % 10000 == 0):
            # print(output)
            # print(list(model.parameters()))
            # print(output.shape)
            # print(loss.item())
        loss_result.append(p)
        weights = model.state_dict()
        tmp = list(weights['fc.weight'])
        cost_result.append(tmp)
    MSE_Loss.append(loss_result)
    costs.append(cost_result)


costs_ = copy.deepcopy(costs)

for i in range(3):
    costs[i][0], costs[i][1], costs[i][2], costs[i][3], costs[i][4], costs[i][5] \
        = costs_[i][3], costs_[i][0], costs_[i][1], costs_[i][5], costs_[i][4], costs_[i][2]

epochs = range(count)

mean = np.mean(MSE_Loss, axis=1)
plt.xlabel('Epoch number')
plt.ylabel('Loss Value')
plt.plot(epochs, mean[0], color='blue', label='β = 0.05')
plt.plot(epochs, mean[1], color='orange', label='β = 0.10')
plt.plot(epochs, mean[2], color='green', label='β = 0.50')
plt.legend()
plt.show()
plt.clf()
#
plt.xlabel('Pair of Leaves')
plt.ylabel('Unlock Value')
X = ["S->v", "V->O", "O->v", "S->O", "O->S", "V->S"]
mean = np.mean(costs, axis=1)
plt.plot(X, mean[0], color='blue', label='β = 0.05')
plt.plot(X, mean[1], color='orange', label='β = 0.10')
plt.plot(X, mean[2], color='green', label='β = 0.50')
plt.legend()
plt.show()
plt.clf()
