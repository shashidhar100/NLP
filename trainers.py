from sklearn.utils import shuffle
import torch
import torch.nn as nn
from tqdm import tqdm


from dataloaders import Dataset
from models import Model
from losses import Loss
from metrics import Metric
from optimizers import Optimizer
from preprocessor import Preprocessor
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,name="LR_twitter"):
        self.name = name
        self.name_map_dic = {
                              "LR_twitter": LR_Twitter  
                            }
    
    def __call__(self,**kwargs):
        return self.name_map_dic[self.name](**kwargs)


class LR_Twitter:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def get_model(self,model_number="0",input_size=2,output_size=1,**kwargs):
        self.model = Model(model_number=model_number)(input_size = input_size,
                                        output_size = output_size,
                                        **kwargs)
        self.model.to(self.device)
        
    def get_data(self,dataset_name="twitter",batch_size=200,shuffle=True,num_workers=0,**kwargs):
        train_data =  Dataset(dataset_name=dataset_name)(train=True,**kwargs)
        test_data =  Dataset(dataset_name=dataset_name)(train=False,**kwargs)
        
        self.trainloader = DataLoader(train_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        self.testloader = DataLoader(test_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        
    def get_optimizer(self,name="sgd",**kwargs):
        self.optimizer = Optimizer(name=name)()(self.model,**kwargs)
    
    def get_loss(self,name="bce",**kwargs):
        self.loss = Loss(name=name)()(**kwargs)
    
    def get_metric(self,name="accuracy",**kwargs):
        self.metric = Metric(name=name)(**kwargs)
                        
    def __call__(self,epochs=100,**kwargs):
        for epoch in range(epochs):
            train_loss = 0.0
            train_accuracy = 0.0
            with tqdm(total=len(self.trainloader),desc="Train Batch: 0",leave=False) as pbar:
                for batch,data in enumerate(self.trainloader):
                    inputs,labels = data[0].to(self.device).float(),data[1].to(self.device).float()
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss(outputs,torch.unsqueeze(labels,axis=1))
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    avg_loss = train_loss/(batch+1)
                    outputs = torch.unsqueeze(torch.Tensor([1.0 if x>0.5 else 0.0 for x in outputs]),axis=1)
                    train_accuracy += self.metric(labels,outputs)
                    avg_accuracy = train_accuracy/(batch+1)
                    pbar.set_postfix_str(s="Train Loss: {} Train Accuracy: {}".format(avg_loss,avg_accuracy))
                    pbar.update(1)
                    pbar.set_description(desc="Train Batch: {}".format(batch+1))
            train_loss = avg_loss
            train_accuracy = avg_accuracy


            with tqdm(total=len(self.testloader),desc="Test Batch: 0",leave=False) as pbar:
                with torch.no_grad():
                    test_loss = 0.0
                    test_accuracy = 0.0
                    for batch,data in enumerate(self.testloader):
                        inputs,labels = data[0].to(self.device).float(),data[1].to(self.device).float()
                        outputs = self.model(inputs)
                        loss = self.loss(outputs,torch.unsqueeze(labels,axis=1))
                        test_loss += loss.item()
                        avg_loss = test_loss/(batch+1)
                        outputs = torch.unsqueeze(torch.Tensor([1.0 if x>0.5 else 0.0 for x in outputs]),axis=1)
                        test_accuracy += self.metric(labels,outputs)
                        avg_accuracy = test_accuracy/(batch+1)
                        pbar.set_postfix_str(s="Test Loss: {} Test Accuracy: {}".format(avg_loss,avg_accuracy))
                        pbar.update(1)
                        pbar.set_description(desc="Test Batch: {}".format(batch+1))
            test_loss = avg_loss
            test_accuracy = avg_accuracy


            print("Epoch {}/{}: 'Train Loss': {} 'Test Loss': {} 'Train Accuracy': {} 'Test Accuracy': {}\n".format(epoch,epochs,train_loss,test_loss,train_accuracy,test_accuracy))
        


if __name__=="__main__":
    trainer = LR_Twitter()
    trainer.get_data()
    trainer.get_model()
    trainer.get_optimizer(lr=0.0001)
    trainer.get_loss()
    trainer.get_metric()
    trainer(epochs=10)
    
               