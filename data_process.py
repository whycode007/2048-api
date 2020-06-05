import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch


class BoardDataset(Dataset):
    """Board dataset."""

    def __init__(self, csv_file,  transform=None):
        """
        :param csv_file: csv文件的路径
        :param transform: 可选的transform
        """

        self.all_data = pd.read_csv(csv_file)
        print(self.all_data.shape)
        self.labels = self.all_data.iloc[:,16]
        self.boards = self.all_data.iloc[:,:16]
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        sample = self.boards.loc[idx].values
        sample = np.expand_dims(sample, axis=0)/11.0
        #print("sample:", sample.shape)
        label = self.labels[idx]

        # if self.transform:
        sample = torch.from_numpy(sample)
        return sample, label

    def all_results(self):
        sample = self.boards.values
        label = self.labels.values
        return sample,label

# 初始化此类
# full_dataset = BoardDataset(csv_file='DATA.csv',transform=transforms.ToTensor())
#     # 绘制前4个图.
#
#
#
# # 划分训练测试数据集方法
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
# print(test_dataset.boards.shape,test_dataset.labels.shape)

#print(test_x.shape,test_y.shape)
# loader = DataLoader(dataset=train_dataset,batch_size=5,shuffle=True)
# for epoch in range(3):
#     for step,(batch_x,batch_y) in enumerate(loader):
#         print(batch_x.shape,batch_y.shape)
#         #torch.Size([5, 16, 1]) torch.Size([5])
# for i in range(len(train_dataset)):
#     sample,label = train_dataset[i]
#     print(sample.shape, sample,type(label),label)
#     #shape:torch.size([1,16,1]) label:int

