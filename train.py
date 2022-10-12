import argparse
import os
import torch
import datasets
import models
import utils
from funcs.vmcr2 import vmcr2_batch_train

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument('--path',help='filepath of the checkpoint',type=str,default=None)
    parser.add_argument('--data_dir',type=str,default='./data',help="directory of datasets")
    parser.add_argument('--batch_size', type=int, required=True, help='batch size for training')
    parser.add_argument('--epoch', type=int, required=True, help='epochs for training')
    parser.add_argument('--latch_epoch', type=int, default=50, help='epochs for latchinging')
    parser.add_argument('--save_epoch', type=int, default=50, help='epochs for checkpoints')
    parser.add_argument('--dataset',type=str,required=True,help='choose dataset')
    parser.add_argument('--out_dim',type=int,default=128,required=True,help='Number of dimensions of the feature')
    parser.add_argument('--num_comps',type=int,default=20,required=True,help='Number of main components for SVD in latching')
    parser.add_argument('--learning_rate',type=float,default=1e-3,help='learning rate of the model')
    parser.add_argument('--log_dir',type=str,default="./log",help='directory for logs')
    parser.add_argument('--metric_mcr2',default=False, action='store_true',help="get the learning curve via mcr2 metric")
    parser.add_argument('--nu_A',type=float,default=2.0,help='nu_A in dictionary update')
    parser.add_argument('--nu_Gamma',type=float,default=5.0,help='nu_Gamma in dictionary update')
    parser.add_argument('--seed',type=int,default=77,help='Random Seed')
    args = parser.parse_args()
    return args

def run_vmcr2(args):
    f_dim=args.out_dim
    qratio=args.num_comps
    assert qratio>0 and qratio<=f_dim,"0<{}<={} is not satisfied.".format(qratio,f_dim)
    trainloader,num_classes=datasets.get_dataloader(
        data_name=args.dataset,data_dir=args.data_dir,train=True,batch_size=args.batch_size
        )
    model=models.get_model(data_name=args.dataset,f_dim=f_dim,use_ce=False)
    pre_history=None
    passed_epoch=-1
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.path is not None:
        weights,pre_history,last_epoch=utils.load_checkpoint(args.path)
        model.load_state_dict(weights)
        passed_epoch=last_epoch
    model=model.train()
    model=model.to(device)
    opt=torch.optim.SGD(model.parameters(),lr=args.learning_rate)
    if torch.cuda.device_count()>1:
        model=torch.nn.DataParallel(model).cuda()
    record_path=os.path.join(
        args.log_dir,"data[{}]_fdim[{}]_ncomps[{}]_epoch[{}]_batch[{}]_lr[{}]_seed[{}]".format(
            args.dataset, f_dim, qratio, args.epoch, args.batch_size, args.learning_rate, args.seed
            ))
    os.makedirs(record_path,exist_ok=True)
    log_path=os.path.join(record_path,'train.log')
    logger=utils.create_logger(log_path)
    model_dir=os.path.join(record_path,'checkpoints')
    params={'dataset':args.dataset,'out_dim':args.out_dim,'num_comps':args.num_comps}
    utils.save_params(record_path,params)
    model,history=vmcr2_batch_train(trainloader,
                              model,
                              args.epoch,
                              args.latch_epoch,
                              num_classes,
                              qratio,
                              device,
                              passed_epoch,
                              logger,
                              True,
                              opt,
                              args.learning_rate,
                              nu_Gamma=args.nu_Gamma,
                              nu_A=args.nu_A,
                              previous_history=pre_history,
                              mcr2_metric=args.metric_mcr2,
                              batch_size=args.batch_size,
                              save_func=utils.save_checkpoint,
                              save_epoch=args.save_epoch,
                              save_path=model_dir)

if __name__ == '__main__':
    args=parse_args()
    torch.random.manual_seed(args.seed)
    run_vmcr2(args)
