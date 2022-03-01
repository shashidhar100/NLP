# from sklearn.utils import shuffle
import torch
import torch.nn as nn
from tqdm import tqdm


from dataloaders import Dataset
from models import Model
import models
from losses import Loss
from metrics import Metric
from optimizers import Optimizer
from preprocessor import Preprocessor
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self,name="Classifier"):
        self.name_map_dic = {
                              "Classifier": Classifier  
                            }
        name_list = list(self.name_map_dic.keys())
        if name not in name_list:
            raise ValueError("Trainer name should be one of these {}".format(name_list))
        self.name = name

    def __call__(self,**kwargs):
        return self.name_map_dic[self.name](**kwargs)

class BaseTrainer:
    def __init__(self,experiment_name=None,**kwargs):
        self.experiment_name = experiment_name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_model(self,name="LR",**kwargs):
        self.model = Model(name=name)(**kwargs)
        self.model.to(self.device)
        
    def get_data(self,dataset_name="twitter",batch_size=200,shuffle=True,num_workers=0,**kwargs):
        train_data =  Dataset(dataset_name=dataset_name)(train=True,**kwargs)
        test_data =  Dataset(dataset_name=dataset_name)(train=False,**kwargs)
        
        self.trainloader = DataLoader(train_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        self.testloader = DataLoader(test_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
        self.dataset_name = dataset_name
        
    def get_optimizer(self,name="sgd",**kwargs):
        self.optimizer = Optimizer(name=name)()(self.model,**kwargs)
    
    def get_loss(self,name="bce",**kwargs):
        self.loss = Loss(name=name)()(**kwargs)
    
    def get_metric(self,name="accuracy",**kwargs):
        self.metric = Metric(name=name)(**kwargs)

    def save_dir(self):
        outputs_dir = os.path.join("..","Outputs")
        if not os.path.exists(outputs_dir):
            os.mkdir(outputs_dir)
        dataset_dir = os.path.join(outputs_dir,self.dataset_name)
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        model_dir = os.path.join(dataset_dir,str(self.model))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        experiments_dir = os.path.join(model_dir,"Experiments")
        if not os.path.exists(experiments_dir):
            os.mkdir(experiments_dir)
        experiemnts_list = os.listdir(experiments_dir)
        if self.experiment_name==None:
            experiment_name = "experiemnt_"+str(len(experiemnts_list))
        experiment_dir = os.path.join(experiments_dir,experiment_name)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        self.save_dir_path = experiment_dir

        


class Classifier(BaseTrainer):
    def __init__(self,**kwargs):
        super(Classifier,self).__init__(**kwargs)
           
                        
    def __call__(self,epochs=100,**kwargs):
        self.save_dir()
        writer = SummaryWriter(self.save_dir_path)
        for epoch in range(1,epochs+1):
            train_loss = 0.0
            train_accuracy = 0.0
            with tqdm(total=len(self.trainloader),desc="Train Batch: 0",leave=False) as pbar:
                for batch,data in enumerate(self.trainloader,1):
                    inputs,labels = data[0].to(self.device).float(),data[1].to(self.device).float()
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    if isinstance(self.model,models.LogisticRegression):
                        labels = torch.unsqueeze(labels,axis=1)
                    loss = self.loss(outputs,labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    avg_loss = train_loss/(batch)
                    if isinstance(self.model,models.LogisticRegression):
                        outputs = torch.unsqueeze(torch.Tensor([1.0 if x>0.5 else 0.0 for x in outputs]),axis=1).to(self.device)
                        labels = torch.squeeze(labels,axis=1)
                    else:
                        _,outputs = torch.max(outputs.data,1)
                        outputs.to(self.device)
                    train_accuracy += self.metric(labels,outputs)
                    avg_accuracy = train_accuracy/(batch)
                    pbar.set_postfix_str(s="Train Loss: {} Train Accuracy: {}".format(avg_loss,avg_accuracy))
                    pbar.update(1)
                    pbar.set_description(desc="Train Batch: {}".format(batch+1))
            train_loss = avg_loss
            train_accuracy = avg_accuracy


            with tqdm(total=len(self.testloader),desc="Test Batch: 0",leave=False) as pbar:
                with torch.no_grad():
                    test_loss = 0.0
                    test_accuracy = 0.0
                    for batch,data in enumerate(self.testloader,1):
                        inputs,labels = data[0].to(self.device).float(),data[1].to(self.device).float()
                        outputs = self.model(inputs)
                        if isinstance(self.model,models.LogisticRegression):
                            labels = torch.unsqueeze(labels,axis=1)
                        loss = self.loss(outputs,labels)
                        test_loss += loss.item()
                        avg_loss = test_loss/(batch)
                        if isinstance(self.model,models.LogisticRegression):
                            outputs = torch.unsqueeze(torch.Tensor([1.0 if x>0.5 else 0.0 for x in outputs]),axis=1).to(self.device)
                            labels = torch.squeeze(labels,axis=1)
                        else:
                            _,outputs = torch.max(outputs.data,1)
                            outputs.to(self.device)
                        test_accuracy += self.metric(labels,outputs)
                        avg_accuracy = test_accuracy/(batch)
                        pbar.set_postfix_str(s="Test Loss: {} Test Accuracy: {}".format(avg_loss,avg_accuracy))
                        pbar.update(1)
                        pbar.set_description(desc="Test Batch: {}".format(batch+1))
            test_loss = avg_loss
            test_accuracy = avg_accuracy

            writer.add_scalar("Train Loss",train_loss,epoch)
            writer.add_scalar("Test Loss",test_loss,epoch)
            writer.add_scalar("Train Accuracy",train_accuracy,epoch)
            writer.add_scalar("Test Accuracy",test_accuracy,epoch)

            print("Epoch {}/{}: 'Train Loss': {} 'Test Loss': {} 'Train Accuracy': {} 'Test Accuracy': {}\n".format(epoch,epochs,train_loss,test_loss,train_accuracy,test_accuracy))
        


if __name__=="__main__":
    trainer = Classifier()
    trainer.get_data()
    trainer.get_model(input_size=2,output_size=1)
    trainer.get_optimizer(lr=0.001)
    trainer.get_loss()
    trainer.get_metric()
    trainer(epochs=10)
    
               