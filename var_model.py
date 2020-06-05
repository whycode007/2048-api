import torch
from torch import nn

# test dataset inputs and only bilstm without cnn
#   Hyper parameters

time_step = 4  # for lstm, there are 4 time steps
input_size = 8  # for lstm, for every time step, the input vector's size is 8
hidden_size_of_lstm = 256
hidden_size_of_cnn = 256
checkpoint_path = 'mymodel2'
biFlag = True


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        if (biFlag):
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_of_lstm,
            num_layers=4, batch_first=True, bidirectional=biFlag)  # (batch_size,time_step,input_size)

        self.fc_lstm = nn.Linear(hidden_size_of_lstm * self.bi_num, hidden_size_of_lstm)
        self.fc = nn.Linear(hidden_size_of_lstm, 4)

    def forward(self, x):
        x = x.float()
        x = x.view(-1, 4, 4)
        x_t = x.transpose(1, 2)
        x_lstm = torch.cat((x, x_t), 2)

        r_out, (h_n, h_c) = self.lstm(x_lstm, None)
        fc1_out = self.fc_lstm(r_out[:, -1, :])

        final_res = self.fc(fc1_out)
        return final_res

# model = Model()
# print(model)





