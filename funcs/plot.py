import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from .util import sort_dataset

def records2arrays(records:list[dict],
                   name_a_0:str="v-mcr2",
                   name_a_1:str="expand",
                   name_a_2:str="compress",
                   name_a_3:str="equ_reg",
                   name_a_4:str="loss_value",
                   name_a_5:str="equ_reg_again",
                   name_b_0:str="mcr2",
                   name_b_1:str="expand",
                   name_b_2:str="compress"
                   )->dict[list]:
    epoches=len(records)
    # records[i]=
    # {
    #   v-mcr2":
    #   {
    #     "expand":expand,
    #     "compress":compress,
    #     "equ_reg":reg,
    #     "loss_value":value,
    #     "equ_reg_again":reg_again
    #   },
    #   "mcr2":
    #   {
    #     "expand":whole_expd,
    #     "compress:":whole_cprs
    #   }
    # }
    vmcr2_expand=list()
    vmcr2_compress=list()
    vmcr2_equ_reg=list()
    vmcr2_loss_value=list()
    vmcr2_equ_reg_again=list()
    mcr2_expand=list()
    mcr2_compress=list()
    for i in range(epoches):
        vmcr2_expand.append(
            records[i][name_a_0][name_a_1])
        vmcr2_compress.append(
            records[i][name_a_0][name_a_2])
        vmcr2_equ_reg.append(
            records[i][name_a_0][name_a_3])
        vmcr2_loss_value.append(
            records[i][name_a_0][name_a_4])
        vmcr2_equ_reg_again.append(
            records[i][name_a_0][name_a_5])
        mcr2_expand.append(
            records[i][name_b_0][name_b_1])
        mcr2_compress.append(
            records[i][name_b_0][name_b_2])
    info={"vmcr2_expand":vmcr2_expand,
          "vmcr2_compress":vmcr2_compress,
          "vmcr2_equ_reg":vmcr2_equ_reg,
          "vmcr2_loss_value":vmcr2_loss_value,
          "vmcr2_equ_reg_again":vmcr2_equ_reg_again,
          "mcr2_expand":mcr2_expand,
          "mcr2_compress":mcr2_compress}
    return info

def plot_loss_mcr(model_dir,mcr2_expand_list,mcr2_compress_list, name="MCR2 loss"):
    loss_expd = np.array(mcr2_expand_list,dtype=float)
    loss_comp = np.array(mcr2_compress_list,dtype=float)
    loss_total = loss_expd-loss_comp
    num_iter = np.arange(len(loss_total))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True)
    ax.plot(num_iter, loss_total, label=r'$\Delta R$', 
                color='green', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, loss_expd, label=r'$R$', 
                color='royalblue', linewidth=1.0, alpha=0.8)
    ax.plot(num_iter, loss_comp, label=r'$R^c$', 
                color='coral', linewidth=1.0, alpha=0.8)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_xlabel('Number of iterations', fontsize=10)
    ax.legend(loc='lower right', prop={"size": 15}, ncol=3, framealpha=0.5)
    fig.tight_layout()
    loss_dir = os.path.join(model_dir, 'figures', 'loss_mcr')
    os.makedirs(loss_dir, exist_ok=True)
    file_name = os.path.join(loss_dir, f'{name}.png')
    plt.savefig(file_name, dpi=500)
    plt.close()
    print("Plot saved to: {}".format(file_name))

def plot_heatmap(model_dir, features, labels, num_classes,name="similarity measure"):
    """Plot heatmap of cosine simliarity for all features. """
    features_sort, _ = sort_dataset(features, labels,classes=num_classes, stack=True)
    sim_mat=torch.mm(features_sort,features_sort.T)
    sim_mat = np.abs(sim_mat.cpu().numpy())
    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True)
    im = ax.imshow(sim_mat, cmap='Blues')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, len(labels), num_classes+1))
    ax.set_yticks(np.linspace(0, len(labels), num_classes+1))
    [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    save_dir = os.path.join(model_dir, 'figures', 'heatmaps')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f"{name}.png")
    plt.savefig(file_name,dpi=500)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_transform(model_dir, inputs, outputs, name):
    fig, ax = plt.subplots(ncols=2)
    inputs = inputs.permute(1, 2, 0)
    outputs = outputs.permute(1, 2, 0)
    outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
    ax[0].imshow(inputs)
    ax[0].set_title('inputs')
    ax[1].imshow(outputs)
    ax[1].set_title('outputs')
    save_dir = os.path.join(model_dir, 'figures', 'images')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'{name}.png')
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_channel_image(model_dir, features, name):
    def normalize(x):
        out = x - x.min()
        out = out / (out.max() - out.min())
        return out
    fig, ax = plt.subplots()
    ax.imshow(normalize(features), cmap='gray')
    save_dir = os.path.join(model_dir, 'figures', 'images')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'{name}.png')
    fig.savefig(file_name)
    print("Plot saved to: {}".format(file_name))
    plt.close()

def plot_nearest_image(model_dir, image, nearest_images, values, name, grid_size=(4, 4)):
    fig, ax = plt.subplots(*grid_size, figsize=(10, 10))
    idx = 1
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if i == 0 and j == 0:
                ax[i, j].imshow(image)
            else:
                ax[i, j].set_title(values[idx-1])
                ax[i, j].imshow(nearest_images[idx-1])
                idx += 1
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.setp(ax[0, 0].spines.values(), color='red', linewidth=2)
    fig.tight_layout()
    
    save_dir = os.path.join(model_dir, 'figures', 'nearest_image')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}.png')
    fig.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()


def plot_image(model_dir, image, name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if image.shape[2] == 1:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    
    save_dir = os.path.join(model_dir, 'figures', 'image')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}.png')
    fig.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

def save_image(image, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if image.shape[2] == 1:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()