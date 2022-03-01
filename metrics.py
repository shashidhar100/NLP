import numpy as np


class Metric:
    def __init__(self,name="accuracy"):
        self.name_map_dic = {
                                "accuracy":Accuracy
                            }
        name_list = list(self.name_map_dic.keys())
        if name not in name_list:
            raise ValueError("Metric name should be one of these {}".format(name_list))
        self.name = name
        
    def __call__(self,**kwargs):
        return self.name_map_dic[self.name](**kwargs)

class Accuracy:
    def __init__(self,**kwargs):
        pass
    
    def __call__(self,true_labels,predicted_labels):
        return (predicted_labels == true_labels).sum().item()/true_labels.size(0)

    def __str__(self):
        return "Accuracy"

    