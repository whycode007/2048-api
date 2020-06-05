import numpy
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import data_process
import var_model

#   Hyper parameters
Epoch = 13
batch_size= 128
time_step = 8 #for lstm, there are 8 time steps
input_size = 4#for lstm, for every time step, the input vector's size is 4
lr = 0.001 #learning rate
train_test=1 #训练测试集比例
hidden_size_of_lstm = 256
checkpoint_path = 'mymodel2'


def load_checkpoint(model, checkpoint_PATH):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH,map_location=torch.device('cpu'))
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
    return model


#载入数据集
train_dataset = data_process.BoardDataset(csv_file='DATA.csv',transform=transforms.ToTensor())
loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)


model = var_model.Model()
print(model)
print('# generator parameters:', sum(param.numel() for param in model.parameters()))
#model = load_checkpoint(model,'mymodel2/epoch10.pth.tar')




optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_func = nn.CrossEntropyLoss() #需要的标签形式为3,4,5...而不是00001这种one hot形式

if torch.cuda.is_available():
    print('yes,gpu')
    model.cuda()
    loss_func.cuda()


for epoch in range(Epoch):
    for step,(x,y) in enumerate(loader):
        b_x = Variable(x)
        b_y = Variable(y)
        if torch.cuda.is_available():
            model.cuda()
            loss_func.cuda()
            b_x,b_y = b_x.cuda(),b_y.cuda()
        output = model(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%100==0:
            print('-----------------------------------------',step//100,'-----------------------------')
            pred = torch.max(output, 1)[1]

            train_correct = (pred == b_y).sum().item()

            print('Epoch: ', epoch, '| train loss: ' , loss,
                  '| train accuracy: ' , train_correct / (batch_size * 1.0))


    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               checkpoint_path + '/epoch' + str(epoch)  + '.pth.tar')