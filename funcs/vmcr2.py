from typing import Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.linalg import norm
from tqdm import tqdm
from .mcr2 import compute_loss

def get_whole_features(data_loader:DataLoader,
                       featurizer:torch.nn.Module,
                       device:Union[torch.device,str]
                       )->tuple[list,list]:
    features=list()
    labels=list()
    for idx,(x_data,y_data) in enumerate(data_loader):
        x_data=x_data.to(device)
        with torch.no_grad():
            Z=featurizer(x_data)
        features.append(Z.cpu())
        labels.append(y_data.cpu())
    all_features=torch.cat(features,dim=0)
    all_labels=torch.cat(labels,dim=0)
    return all_features,all_labels

def get_repre_matrices_(Z:Tensor,
                        y:Tensor,
                        num_classes:int,
                        unit:bool=True
                        )->list[Tensor]:
    _,d=Z.shape
    repre_each_class=[torch.zeros(size=(d,d),device=Z.device)]*num_classes
    if not unit:
        for j in range(num_classes):
            ztheta_inclass=Z[y==j].T
            if len(ztheta_inclass.shape)==2:
                repre_each_class[j]=torch.mm(ztheta_inclass,ztheta_inclass.T)
    else:
        num_eachclass=[0]*num_classes
        for j in range(num_classes):
            num_eachclass[j]=sum(y==j).item()
            if num_eachclass[j]!=0:
                ztheta_inclass=Z[y==j].T
                repre_each_class[j]=torch.mm(ztheta_inclass,ztheta_inclass.T)
                if num_eachclass[j]!=0:
                    repre_each_class[j]/=num_eachclass[j]
    return repre_each_class

def get_repre_each_class(dataloader:DataLoader,
                         featurizer:torch.nn.Module,
                         num_classes:int,
                         device:Union[torch.device,str],
                         create_grad_graph:bool=True,
                         verbose:bool=True,
                         v_desc:str="Extracting feature representations"
                         )->list[Tensor]:
    assert isinstance(num_classes,int) and num_classes>0,"num_classes={}>0 is required!".format(num_classes)
    pbar=tqdm(iterable=dataloader,
              total=len(dataloader),
              disable=(not verbose),
              desc=v_desc)
    num_eachclass=[0]*num_classes
    if create_grad_graph:
        for idx,(x_data,y_data) in enumerate(pbar):
            x_data=x_data.to(device)
            y_data=y_data.to(device)
            Z=featurizer(x_data)
            batch_eachclass=get_repre_matrices_(Z,y_data,num_classes,unit=False)
            if idx==0:
                for j in range(num_classes):
                    num_eachclass[j]+=(sum(y_data==j).item())
                eachclass=batch_eachclass
            else:
                for j in range(num_classes):
                    num_eachclass[j]+=(sum(y_data==j).item())
                    eachclass[j]+=batch_eachclass[j]
    else:
        with torch.no_grad():
            for idx,(x_data,y_data) in enumerate(pbar):
                x_data=x_data.to(device)
                y_data=y_data.to(device)
                Z=featurizer(x_data)
                batch_eachclass=get_repre_matrices_(Z,y_data,num_classes,unit=False)
                if idx==0:
                    for j in range(num_classes):
                        num_eachclass[j]+=(sum(y_data==j).item())
                    eachclass=batch_eachclass
                else:
                    for j in range(num_classes):
                        num_eachclass[j]+=(sum(y_data==j).item())
                        eachclass[j]+=batch_eachclass[j]
    for j in range(num_classes):
        if num_eachclass[j]!=0:
            eachclass[j]/=num_eachclass[j]
    return eachclass

def get_each_classnum(dataloader:DataLoader,num_classes:int)->list[int]:
    each_classnum=[0]*num_classes
    for _,(_,y_data) in enumerate(dataloader):
        for j in range(num_classes):
            each_classnum[j]+=sum((y_data==j)).item()
    return each_classnum

def latcher(repre_matrices:list[Tensor],
            s:int,
            device:Union[torch.device,str]='cpu'
            )->tuple[Tensor,Tensor]:
    k=len(repre_matrices)
    d,_=repre_matrices[0].shape
    assert isinstance(s,int),"s={} must be an integer.".format(s)
    assert s>0 and s<=d,"s={} does not satisfy 0<s<=d, where d={}".format(s,d)
    q=int(s*k)
    A=torch.zeros(size=(q,k),device=device)
    Gamma=torch.zeros(size=(d,q),device=device)
    for j in range(k):
        U,S,_=torch.linalg.svd(repre_matrices[j]) #U:d*d,S:d
        Gamma[:,j*s:(j+1)*s]=U[:,0:s] #d*(k*s)=d*q: unit columns
        A[j*s:(j+1)*s,j]=S[0:s] #(k*s)*k=q*k: a sparse block matrix with non-negative elements
    A=A.requires_grad_(True)
    Gamma=Gamma.requires_grad_(True)
    return A,Gamma

def latching_scaled(dataloader:DataLoader,
                    featurizer:torch.nn.Module,
                    num_class:int,
                    s:int,
                    device:Union[torch.device,str]='cpu',
                    verbose:bool=False
                    )->tuple[Tensor,Tensor]:
    v_desc="Extracting feature representations for latching"
    repre_matrices=get_repre_each_class(dataloader,featurizer,num_class,device,False,verbose,v_desc)
    return latcher(repre_matrices,s,device)

def vdeltar(Gamma:Tensor,
            A:Tensor,
            each_classnum:list[int],
            epsilon:float=0.5
            )->tuple[Tensor,Tensor,list[Tensor]]:
    d,q=Gamma.shape
    _,k=A.shape
    assert len(each_classnum)==k,"A.size(1)==len(each_classnum) must be satisfied!"
    device=Gamma.device
    I=torch.eye(n=d,device=device)
    m=sum(each_classnum)
    ec=torch.tensor(each_classnum)
    alp_j=d/(epsilon)
    gam_js=1.0*ec/m
    s=q//k
    # To accumulate \Gamma\Diag{A_j}\Gamma^T
    expand_core=torch.zeros(size=(d,d),device=device)
    # To accumulate \gamma_j(\sum_{l=1}^{q}\log{1_\alpha_j A_{l,j}})
    compress_core=torch.tensor(0.0,device=device)
    # store \Gamma\Diag{A_j}\Gamma^T by index j
    sparse_matrices=list()
    for j in range(k):
        Gamma_j=Gamma[:,j*s:(j+1)*s]
        A_j=torch.diag(A[j*s:(j+1)*s,j])
        sparse_matrix=torch.mm(
            torch.mm(
                Gamma_j,A_j
            ),Gamma_j.T
        )
        sparse_matrices.append(sparse_matrix)
        expand_core+=sparse_matrix

        compress_item=torch.tensor(0.0,device=device)
        for el in range(j*s,(j+1)*s,1):
            compress_item+=torch.log(1+alp_j*A[el,j])
        compress_core+=gam_js[j]*compress_item
    
    expand=torch.logdet(I+d/(epsilon)*expand_core)/2.0
    compress=compress_core/2.0
    return expand,compress,sparse_matrices

def constraint_equ(repre_matrices:list[Tensor],
                   dict_matrices:list[Tensor],
                   each_classnum:list[int],
                   grad_features:bool=False,
                   start_scale:bool=False,
                   current_progress:float=0.0,
                   threshold_progress:float=0.1)->Tensor:
    k=len(each_classnum)
    m=sum(each_classnum)
    ec=torch.tensor(each_classnum)
    gam_js=1.0*ec/m
    assert k==len(repre_matrices) and k==len(dict_matrices),"Incompatible class quantities!"
    device=repre_matrices[0].device
    res=torch.tensor(0.0,device=device)
    for j in range(k):
        if grad_features:
            repre_m=repre_matrices[j]
            dict_m=dict_matrices[j].clone().detach()
        else:
            repre_m=repre_matrices[j].clone().detach()
            dict_m=dict_matrices[j]
        if start_scale:
            g_k=(1-(gam_js[j]+1e-8))
            g_b=(gam_js[j]+1e-8)
            if current_progress<threshold_progress:
                gamj_scaled=g_k*(current_progress/threshold_progress)+g_b
            else:
                gamj_scaled=1.0
        else:
            gamj_scaled=(gam_js[j]+1e-8)
        res+=(norm(repre_m-dict_m,ord='fro')**2)/gamj_scaled
    return res

def total_loss(repre_matrices:list[Tensor],
               Gamma:Tensor,
               A:Tensor,
               each_classnum:list[int],
               epsilon:float=0.5,
               mu:float=1.0,
               start_scale:bool=False,
               current_progress:float=0.0,
               threshold_progress:float=0.1
               )->tuple[Tensor,list[Tensor],float,float,float]:
    k=A.size(1)
    #m=sum(each_classnum)
    expand,compress,sparse_matrices=vdeltar(Gamma,A,each_classnum,epsilon)
    loss_reg=constraint_equ(repre_matrices,sparse_matrices,each_classnum,False,
                            start_scale,current_progress,threshold_progress)
    if start_scale:
        if current_progress<threshold_progress:
            k_scaled=k/(1+(k-1)*(current_progress/threshold_progress))
        else:
            k_scaled=1.0
    else:
        k_scaled=k
    value=expand-compress-0.5*mu*loss_reg/(k*k_scaled)
    return value,sparse_matrices,expand.item(),compress.item(),loss_reg.item()

def grad_upper_bounds(repre_matrices:list[Tensor],
                      Gamma:Tensor,
                      A:Tensor,
                      mu:float=1.0
                      )->tuple[Tensor,Tensor]:
    q=Gamma.size(1)
    k=A.size(1)
    s=q//k
    device=Gamma.device
    L_Gam=torch.tensor(0.0,device=device)
    with torch.no_grad():
        for j in range(k):
            Aj_inf=norm(A[j*s:(j+1)*s,j],ord=float('inf'))
            L_Gam_item=(norm(repre_matrices[j],ord='fro')+Aj_inf)*Aj_inf
            L_Gam+=L_Gam_item
        L_Gam*=2.0*mu/k

        Gam_squared=torch.mm(Gamma.T,Gamma)**2 #q*q
        L_A=norm(Gam_squared,ord='fro')*mu/k
    
    return L_Gam,L_A

def update_dictionary(value:Tensor,
                      Gamma:Tensor,
                      A:Tensor,
                      L_Gamma:Tensor=torch.tensor(1.0),
                      L_A:Tensor=torch.tensor(1.0),
                      nu_Gamma:float=5.0,
                      nu_A:float=5.0,
                      p_A:float=1.0)->tuple[Tensor,Tensor]:
    q,k=A.shape
    s=q//k
    L_Gamma=L_Gamma.to(Gamma.device)
    L_A=L_A.to(A.device)
    delta_Gamma=torch.autograd.grad(outputs=value,inputs=Gamma,retain_graph=True)[0]
    delta_A=torch.autograd.grad(outputs=value,inputs=A,retain_graph=True)[0]
    with torch.no_grad():
        Gamma_new=Gamma+nu_Gamma/L_Gamma*delta_Gamma
        A_new=A+nu_A/L_A*delta_A
    A_new=torch.nn.functional.relu(A_new)
    if p_A>0:
        for j in range(k):
            A_new[j*s:(j+1)*s,j]=torch.nn.functional.normalize(A_new[j*s:(j+1)*s,j],p=p_A,dim=0)
    A_new=A_new.requires_grad_(True)
    Gamma_new=torch.nn.functional.normalize(Gamma_new,p=2,dim=0).requires_grad_(True)
    return Gamma_new,A_new

def epoch_updater_scaled(dataloader:DataLoader,
                         featurizer:torch.nn.Module,
                         optimizer:torch.optim.Optimizer,
                         Gamma:Tensor,
                         A:Tensor,
                         num_classes:int,
                         device:Union[torch.device,str]='cpu',
                         epsilon:float=0.5,
                         mu:float=1.0,
                         nu_Gamma:float=5.0,
                         nu_A:float=5.0,
                         verbose:bool=True,
                         start_scale:bool=False,
                         current_progress:float=0.0,
                         threshold_progress:float=0.1,
                         p_A_mode:float=0.0,
                         p_A_progress:float=0.0
                         )->tuple[torch.nn.Module,Tensor,Tensor,float,float,float,float]:
    v_desc="Extracting feature representations for training"
    d,q=Gamma.shape
    _,k=A.shape
    s=q//k
    avg_expd=0.0
    avg_cprs=0.0
    avg_reg=0.0
    avg_reg_again=0.0
    avg_value=0.0
    pbar=tqdm(iterable=dataloader,
              total=len(dataloader),
              disable=(not verbose),
              desc=v_desc)
    for idx,(x_data,y_data) in enumerate(pbar):
        each_classnum=[0]*num_classes
        for j in range(num_classes):
            each_classnum[j]=sum(y_data==j).item()
        optimizer.zero_grad()
        x_data=x_data.to(device)
        y_data=y_data.to(device)
        Z=featurizer(x_data)
        repre_matrices=get_repre_matrices_(Z,y_data,num_classes,unit=True)
        value,_,expand,compress,reg=total_loss(
            repre_matrices,Gamma,A,each_classnum,epsilon,mu,start_scale,current_progress,threshold_progress)
        avg_expd+=expand
        avg_cprs+=compress
        avg_reg+=reg
        avg_value+=value.item()
        L_Gamma,L_A=grad_upper_bounds(repre_matrices,Gamma,A,mu)
        if p_A_mode==0.0:
            p_A=3
        elif p_A_mode==1.0:
            p_A=2
        else:
            p_A=3-p_A_progress
        Gamma,A=update_dictionary(value,Gamma,A,L_Gamma,L_A,nu_Gamma,nu_A,p_A)
        sparse_matrices=list()
        with torch.no_grad():
            for j in range(k):
                Gamma_j=Gamma[:,j*s:(j+1)*s]
                A_j=torch.diag(A[j*s:(j+1)*s,j])
                sparse_matrix=torch.mm(
                    torch.mm(
                        Gamma_j,A_j
                    ),Gamma_j.T
                )
                sparse_matrices.append(sparse_matrix)
        loss_reg=constraint_equ(repre_matrices,sparse_matrices,each_classnum,True,start_scale,current_progress,threshold_progress)
        loss_reg.backward()
        avg_reg_again+=loss_reg.item()
        optimizer.step()
    avg_expd/=len(dataloader)
    avg_cprs/=len(dataloader)
    avg_reg/=len(dataloader)
    avg_reg_again/=len(dataloader)
    avg_value/=len(dataloader)
    return featurizer,Gamma,A,avg_expd,avg_cprs,avg_reg,avg_reg_again,avg_value

def get_mcr2_loss(Z:torch.Tensor,
                  y:torch.Tensor,
                  num_classes:int,
                  epsilon:float=0.5,
                  require_grad:bool=False)->tuple[torch.Tensor,torch.Tensor]:
    if require_grad:
        _,expand,compress=compute_loss(Z,y,num_classes,epsilon)
    else:
        with torch.no_grad():
            _,expand,compress=compute_loss(Z,y,num_classes,epsilon)
    return expand,compress

def get_whole_mcr2_loss(trainloader:DataLoader,
                        featurizer:torch.nn.Module,
                        num_classes:int,
                        device='cpu',
                        epsilon=0.5
                        )->tuple[float,float]:
    Z,labels=get_whole_features(trainloader,featurizer,device)
    expand,compress=get_mcr2_loss(Z,labels,num_classes,epsilon,False)
    return expand.item(),compress.item()

def vmcr2_batch_train(trainloader,
                      featurizer,
                      num_epoch,
                      latch_epoch,
                      num_classes,
                      q_ratio,
                      device,
                      passed_epoch=-1,
                      logger=None,
                      verbose_train=True,
                      optimizer=None,
                      nu_theta=1e-3,
                      nu_Gamma=5.0,
                      nu_A=5.0,
                      mu=1.0,
                      epsilon=0.5,
                      previous_history=None,
                      verbose_latch=False,
                      mcr2_metric=False,
                      batch_size=1000,
                      save_func=None,
                      save_epoch=100,
                      save_path=None,
                      warmup_persentage=1.0/20,
                      change_equ_lr=False,
                      accelerate_persentage=1.0/2,
                      pAswitch_persentage=3.0/5):
    if optimizer is None: # if optimizer is not specified, then assign SGD
        if isinstance(featurizer,torch.nn.DataParallel):
            params=featurizer.module.parameters()
        else:
            params=featurizer.parameters()
        optimizer=torch.optim.SGD(params,lr=nu_theta)
    featurizer=featurizer.to(device)
    assert warmup_persentage>0 and accelerate_persentage>0 and (warmup_persentage+accelerate_persentage)<1
    assert pAswitch_persentage>=warmup_persentage
    if logger is not None:
        func_print=logger.info
    else:
        func_print=print
    func_print("Start Training...")
    #each_classnum=get_each_classnum(trainloader,num_classes)
    previous_weights=featurizer.module.state_dict() if isinstance(featurizer,torch.nn.DataParallel) else featurizer.state_dict()
    previous_epoch=passed_epoch
    if previous_history is not None:
        loss_history=previous_history
    else:
        loss_history=list()
    func_print("Initialize the dictionary by latching...")
    eps_k=epsilon*(2*batch_size/num_classes-1)/(num_epoch*warmup_persentage)
    eps_b=epsilon*2*batch_size/num_classes
    try:
        A,Gamma=latching_scaled(trainloader,featurizer,num_classes,q_ratio,device,verbose_latch)
        for current_epoch in range((passed_epoch+1),num_epoch):
            if current_epoch/num_epoch<warmup_persentage:
                epsilon_scaled=-eps_k*current_epoch+eps_b
                p_A_progress=0
                p_A_switch=0
            else:
                epsilon_scaled=epsilon
                p_A_progress=min((current_epoch/num_epoch-warmup_persentage)/(pAswitch_persentage-warmup_persentage),1)
                p_A_switch=p_A_progress
            start_scale=(current_epoch>=(num_epoch*warmup_persentage)) and change_equ_lr
            scale_progress=(current_epoch-num_epoch*warmup_persentage)/num_epoch
            featurizer,Gamma,A,expand,compress,reg,reg_again,value=epoch_updater_scaled(
                trainloader,featurizer,optimizer,Gamma,A,
                num_classes,device,epsilon_scaled,mu,nu_Gamma,nu_A,
                verbose_train,start_scale,scale_progress,accelerate_persentage,p_A_switch,p_A_progress)
            record={"v-mcr2":{"expand":expand,"compress":compress,"equ_reg":reg,"loss_value":value,"equ_reg_again":reg_again}}
            func_print("Epoch {}/{}: expand={:.2f}, compress={:.2f}, equ_reg={:.2f}, loss_value={:.2f}, equ_reg_again={:.2f}".format(
                current_epoch,num_epoch,expand,compress,reg,value,reg_again
            ))
            if mcr2_metric:
                whole_expd,whole_cprs=get_whole_mcr2_loss(trainloader,featurizer,num_classes,device,epsilon)
                record.update({"mcr2":{"expand":whole_expd,"compress":whole_cprs}})
                func_print("Epoch {}/{} using original MCR2: expand={:.3f}, compress={:.3f}, total={:.3f}".format(
                    current_epoch,num_epoch,whole_expd,whole_cprs,whole_expd-whole_cprs
                ))
            loss_history.append(record)
            previous_weights=featurizer.module.state_dict() if isinstance(featurizer,torch.nn.DataParallel) else featurizer.state_dict()
            previous_epoch=current_epoch
            if current_epoch%latch_epoch==0:
                func_print("Latching again at Epoch {}.".format(current_epoch))
                A,Gamma=latching_scaled(trainloader,featurizer,num_classes,q_ratio,device,verbose_latch)
            if (save_func is not None) and (save_path is not None):
                if current_epoch%save_epoch==0:
                    func_print("Save checkpoints at Epoch {}.".format(current_epoch))
                    save_func(save_path,previous_weights,previous_epoch,loss_history)
    except KeyboardInterrupt:
        func_print("KeyboardInterrupt")
        if (save_func is not None) and (save_path is not None):
            func_print("Save checkpoints due to Keyboard Interrupt")
            save_func(save_path,previous_weights,previous_epoch,loss_history)
    except RuntimeError as re:
        func_print(str(re))
        if "out of memory" in str(re):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise re
    except Exception as e:
        func_print(str(e))
        raise e
    else:
        func_print("Training finished.")
        if (save_func is not None) and (save_path is not None):
            func_print("Save checkpoints of the final model.")
            save_func(save_path,previous_weights,previous_epoch,loss_history)
    
    return featurizer,loss_history
