from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F


class Model:
    def __init__(self,name="LR"):
        self.name_map_dic = {
                                "LR":LogisticRegression,
                                }
        name_list = list(self.name_map_dic.keys())
        if name not in name_list:
            raise ValueError("Model name should be one of these {}".format(name_list))
        self.name = name
    
    def __call__(self,**kwargs):
        return self.name_map_dic[self.name](**kwargs)



class LogisticRegression(nn.Module):
    def __init__(self,input_size,output_size,**kwargs):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(input_size,output_size)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
    
    def forward(self,x):
        return torch.sigmoid(self.linear(x))
    
    def __str__(self):
        return "Logistic Regression Model"

lr = LogisticRegression(2,1)
print(lr)