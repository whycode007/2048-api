data_process.py用以进行数据的处理，加载训练集
var_model.py 是用到的bilstm模型
train.py是用来进行模型训练，并生成checkpoint
agents.py中写出了使用监督学习生成的模型的agent，模型导入和走下一步的判断均在此文件中
数据集和生成的模型因为超过了25mb未进行上传，可以联系immernoch@sjtu.edu.cn进行传递
运行evaluate.py即可进行测试