import torch
import torch.nn as nn
from opt_einsum import contract

def expand_volumn(V,eps=0.1):
    m=V.size(0)
    C=V.size(1)
    
    I = torch.eye(C)
    for _ in range(len(V.size())-2):
        I=I.unsqueeze(-1)
    I = I.to(V.device)
    
    alpha = C / (m * eps)
    exp_v = I + alpha * contract('ji...,jk...->ik...', V, V.conj())
    return exp_v,alpha

def compute_inv(m,alpha):
    m_inv=alpha*contract('...ij->ij...',torch.inverse(contract('ij...->...ij',m)))
    return m_inv

def compress_volumn(V,y,num_classes,eps=0.1):
    C=V.size(1)
    num_classes=int(num_classes)
    I = torch.eye(C)
    for _ in range(len(V.size())-2):
        I=I.unsqueeze(-1)
    I = I.to(V.device)
    
    Com_ms=list()
    Alphas=list()
    Gammas=list()

    for j in range(num_classes):
        V_j = V[(y == int(j))].clone()
        gam_j=V_j.shape[0]
        Alp_j = C / ((gam_j+1e-8) * eps)
        Gammas.append(gam_j)
        cmp=I+Alp_j*contract('ji...,jk...->ik...', V_j, V_j.conj())
        Alphas.append(Alp_j)
        Com_ms.append(cmp)
    return Com_ms,Alphas,Gammas

def compute_multi_inv(Ms,Aas):
    Cs=torch.zeros(Ms.size(),dtype=Ms.dtype).to(Ms.device)
    for j in range(len(Aas)):
        Cs[j]=compute_inv(Ms[j],Aas[j])
    return Cs

def covariance(X):
    return contract('ji...,jk...->ik...', X, X.conj())

def logdet(X):
    sgn, logdet = torch.linalg.slogdet(X)
    return sgn * logdet

def compute_discrimn_loss(V,eps=0.1):
    assert len(V.size())>=2,"Dimensions of input batch must at least 2"
    last_dims=1
    for i in torch.tensor(V.shape[2:]).tolist():
        last_dims*=i
    cov,_=expand_volumn(V,eps)
    loss_expd = logdet(contract('ij...->...ij',cov)).sum() / (2 * last_dims)
    return loss_expd.real,last_dims

def compute_compress_loss(V,y,num_classes,eps=0.1,last_dims=None):
    assert len(V.size())>=2,"Dimensions of input batch must at least 2"
    if last_dims is None:
        last_dims=1
        for i in torch.tensor(V.shape[2:]).tolist():
            last_dims*=i
    
    label=torch.squeeze(y)
    assert len(label.size())==1,"Incompatible shape of label batch"

    compjs,_,gammas=compress_volumn(V, label,num_classes, eps)

    scalars=list()
    compress_loss=list()

    for j in range(num_classes):
        sca_j = torch.tensor(gammas[j] / (2*label.size(0)))
        comp_j = logdet(contract('ij...->...ij',compjs[j].real)).sum() / last_dims
        scalars.append(sca_j)
        compress_loss.append(comp_j)
    
    return compress_loss,scalars

def compute_loss(V, y,num_classes, eps=0.5):

    loss_expd,last_dims=compute_discrimn_loss(V,eps)
    compress_loss,scalars=compute_compress_loss(V,y,num_classes,eps,last_dims)

    loss_comp=0.0
    for c,s in zip(compress_loss,scalars):
        loss_comp+=c*s
    
    return loss_comp - loss_expd, loss_expd, loss_comp

class mcr2(nn.Module):
    def __init__(self, num_class=None,eps=0.5, gam=1.):
        super(mcr2, self).__init__()
        self.eps = eps
        self.gam = gam
        self.num_class=num_class
    
    def forward(self, Z, y):
        if self.num_class is None:
            self.num_class=len(torch.squeeze(y).unique())
            print("Warning: `num_class` is adjusted to the number of unique values in the label batch.")
        _,R,Rc=compute_loss(Z,y,self.num_class,self.eps)
        return self.gam*Rc-R #minimization
    
    def set_numclass(self,num_class):
        self.num_class=num_class
    
    def get_numclass(self):
        return self.num_class
