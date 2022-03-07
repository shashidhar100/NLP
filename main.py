import logging
from trainers import Trainer
import argparse

logger = logging.getLogger()


def main(args):
    Trainer(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--trainer",type=str,help="Name of the trainer (default = 'classifier')",default="classifier")
    parser.add_argument("-m","--model_name",type=str,help="Name of the model (default = 'LR')",default="LR")
    parser.add_argument("-d","--dataset_name",type=str,help="Name of the dataset (default = 'twitter'",default='twitter')
    
