import torch
import os
import argparse
import datasets
import utils
import models
from utils import get_whole_features,load_params,save_params
from funcs import evaluate,plot

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument('--path',help='filepath of the checkpoint',type=str,default=None)
    parser.add_argument('--data_dir',type=str,default='./data',help="directory of datasets")
    parser.add_argument('--batch_size', type=int, required=True, help='batch size for evaluation')
    parser.add_argument('--n_comps', type=int, default=0, help='# of components for evaluation')
    parser.add_argument('--seed',type=int,default=77,help='Random Seed')
    args = parser.parse_args()
    return args

def get_evaluation(args):
    model_dir=os.path.dirname(os.path.dirname(args.path))
    eval_dir=os.path.join(model_dir,"evaluation")
    os.makedirs(eval_dir,exist_ok=True)
    model_info=load_params(model_dir)
    dataset=model_info["dataset"]
    num_comps=model_info["num_comps"] if args.n_comps<1 else args.n_comps
    out_dim=model_info["out_dim"]
    trainloader,num_classes=datasets.get_dataloader(data_name=dataset,data_dir=args.data_dir,train=True,batch_size=args.batch_size)
    testloader,_=datasets.get_dataloader(data_name=dataset,data_dir=args.data_dir,train=False,batch_size=args.batch_size)
    model_dicts,history,epoch=utils.load_checkpoint(args.path)
    model=models.get_model(data_name=dataset,f_dim=out_dim,use_ce=False)
    model.load_state_dict(model_dicts)
    model=model.eval()
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_features,train_labels=get_whole_features(trainloader,model,device,verbose=True,desc="Extracting training features.")
    test_features,test_labels=get_whole_features(testloader,model,device,verbose=True,desc="Extracting testing features.")
    train_acc, test_acc=evaluate.nearsub(train_features,train_labels,test_features,test_labels,num_classes,num_comps,False)
    print('SVD: train={}, test={}'.format(train_acc, test_acc))
    acc_dict = {'train': train_acc, 'test': test_acc}
    save_params(eval_dir, acc_dict, name='acc_SVD')
    hist_arrays=plot.records2arrays(history)
    print("Getting the plot of MCR2 learning curve...")
    plot.plot_loss_mcr(eval_dir,hist_arrays["mcr2_expand"],hist_arrays["mcr2_compress"],name="MCR2 Learning Curve")
    print("Getting the heatmap of similarities in test features...")
    plot.plot_heatmap(eval_dir,test_features,test_labels,num_classes,name="Similarity of features in test data")
    print("Getting the heatmap of similarities in random {} train features...".format(len(test_labels)))
    plot.plot_heatmap(
        eval_dir,
        train_features[:len(test_labels)],
        train_labels[:len(test_labels)],
        num_classes,
        name="Similarity of features in {} training data".format(len(test_labels)))

if __name__ == '__main__':
    args=parse_args()
    torch.random.manual_seed(args.seed)
    get_evaluation(args)
