from turtle import forward
import torch.nn as nn
import torch


class Model:
    def __init__(self,model_number="0"):
        self.model_number = model_number
        self.model_map_dic = {
                                "0":LogisticRegression
                                }
    
    def __call__(self,**kwargs):
        return self.model_map_dic[self.model_number](**kwargs)
    

class LogisticRegression(nn.Module):
    def __init__(self,input_size,output_size,**kwargs):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(input_size,output_size)
    
    def forward(self,x):
        return torch.sigmoid(self.linear(x))
    
    def __str__(self):
        return "Logistic Regression Model"
        