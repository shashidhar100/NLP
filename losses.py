import torch.nn as nn


class Loss:
    def __init__(self,name="cross_entropy"):
        self.name_map_dic = {
                                "cross_entropy":CrossEntropy,
                                "bce":BinaryCrossEntropy
                            }
        name_list = list(self.name_map_dic.keys())
        if name not in name_list:
            raise ValueError("Loss name should be one of these {}".format(name_list))
        self.name = name
        
    
    def __call__(self,**kwargs):
        return self.name_map_dic[self.name](**kwargs)

class CrossEntropy:
    def __init__(self,**kwargs):
        pass
    
    def __call__(self,**kwargs):
        return nn.CrossEntropyLoss(**kwargs)

    def __str__(self):
        return "Cross Entropy"
    
class BinaryCrossEntropy:
    def __init__(self,**kwargs):
        pass
    
    def __call__(self,**kwargs):
        return nn.BCELoss(**kwargs)

    def __str__(self):
        return "Binary Cross Entropy"



