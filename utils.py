import torch
import os
import logging
import json
from typing import Dict,Any
from tqdm import tqdm

def save_checkpoint(path_dir:str,model_weights:Dict[str, Any],current_epoch:int,history:list,name:str='model')->bool:
    os.makedirs(path_dir,exist_ok=True)
    filename=os.path.join(path_dir,"{}_epoch{}.pth".format(name,current_epoch))
    info={"model":model_weights,"history":history,"epoch":current_epoch}
    torch.save(info,filename)
    return True

def load_checkpoint(filepath:str)->tuple[Dict[str, Any],list,int]:
    info=torch.load(filepath)
    return (info['model'],info['history'],info['epoch'])

def get_whole_features(data_loader,featurizer,device,verbose=True,desc="extracting features"):
    features=list()
    labels=list()
    featurizer=featurizer.to(device)
    pbar=tqdm(iterable=data_loader,total=len(data_loader),disable=(not verbose),desc=desc)
    for idx,(x_data,y_data) in enumerate(pbar):
        x_data=x_data.to(device)
        with torch.no_grad():
            Z=featurizer(x_data)
        features.append(Z.cpu())
        labels.append(y_data.cpu())
    all_features=torch.cat(features,dim=0)
    all_labels=torch.cat(labels,dim=0)
    return all_features,all_labels

def create_logger(log_path):
    """
    Export log to file and console.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # a handler to write log file
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # a handler to output log to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def save_params(model_dir, params, name='params', name_prefix=None):
    """Save params to a .json file. Params is a dictionary of parameters."""
    if name_prefix:
        model_dir = os.path.join(model_dir, name_prefix)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def load_params(model_dir,name="params.json"):
    """Load params.json file in model directory and return dictionary."""
    path = os.path.join(model_dir, name)
    with open(path, 'r') as f:
        _dict = json.load(f)
    return _dict
