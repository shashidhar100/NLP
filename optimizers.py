import torch.optim as optim


class Optimizer:
    def __init__(self,name="sgd"):
        self.name = name
        self.name_map_dic = {
                                "sgd":SGD
                            }
    
    def __call__(self,**kwargs):
        return self.name_map_dic[self.name](**kwargs)

class SGD:
    def __init__(self):
        pass

    def __call__(self,model,**kwargs):
        return optim.SGD(model.parameters(),**kwargs)

    def __str__(self):
        return "SGD"