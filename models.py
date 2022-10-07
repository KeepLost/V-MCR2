import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Type, Union

class simple_net(nn.Module):
    def __init__(self,d:int,get_logits:bool=False,num_class:int=10) -> None:
        super().__init__()
        self.d=d
        self.get_logits=get_logits
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=None)
        self.fc1=nn.Linear(in_features=12544,out_features=d)
        self.fc2=nn.Linear(in_features=d,out_features=d)
        if self.get_logits:
            self.fc3=nn.Linear(in_features=d,out_features=num_class)
    
    def forward(self,x:Tensor) -> Tensor:
        out=nn.ReLU()(self.conv1(x))
        out=nn.ReLU()(self.conv2(out))
        out=nn.Dropout(p=0.25)(self.pool(out))
        out=nn.Flatten()(out)
        out=nn.ReLU()(self.fc1(out))
        out=self.fc2(nn.Dropout(p=0.5)(out))
        out=F.normalize(out,p=2,dim=1)
        if self.get_logits:
            out=self.fc3(out)
        return out

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: BasicBlock,
        layers: List[int],
        dims:int=200,
        num_classes: int = 100,
        get_logits: bool=False,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dims=dims
        self.get_logits=get_logits
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 512,bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512,self.dims,bias=True)
        if self.get_logits:
            self.fc3=nn.Linear(self.dims,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                # if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                #     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                # elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                #     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = F.normalize(x,p=2,dim=1)
        if self.get_logits:
            x=self.fc3(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet(
    block: BasicBlock,
    layers: List[int],
    dims: int,
    num_classes: int,
    use_ce: bool=False,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers,dims,num_classes,use_ce, **kwargs)
    return model

def resnet18(dims: int,num_classes: int, use_ce: bool=False,**kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], dims, num_classes,use_ce, **kwargs)

def get_model(data_name:str,f_dim:int=128,use_ce:bool=False)->Union[simple_net,ResNet]:
    if data_name=='mnist':
        return simple_net(d=f_dim,get_logits=use_ce,num_class=10)
    elif data_name=='cifar10':
        return resnet18(dims=f_dim,num_classes=10,use_ce=use_ce)
    elif data_name=='cifar100':
        return resnet18(dims=f_dim,num_classes=100,use_ce=use_ce)
    elif data_name=='tiny_imagenet':
        return resnet18(dims=f_dim,num_classes=200,use_ce=use_ce)
    else:
        raise ValueError(f"The model for {data_name} is not implemented!")
