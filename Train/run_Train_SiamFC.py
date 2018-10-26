"""
PyTorch implementation of SiamFC (Luca Bertinetto, et al., ECCVW, 2016)
Written by Heng Fan
"""

from SiamNet import *
from VIDDataset import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from Config import *
from Utils import *
import torchvision.transforms as transforms
from DataAugmentation import *
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# fix random seed
np.random.seed(1357)
torch.manual_seed(1234)


def train(data_dir, train_imdb, val_imdb, model_save_path="./model/", use_gpu=True):

    # initialize training configuration
    config = Config()

    # do data augmentation in PyTorch;
    # you can also do complex data augmentation as in the original paper
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride

    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # load data (see details in VIDDataset.py)
    train_dataset = VIDDataset(train_imdb, data_dir, config, train_z_transforms, train_x_transforms)
    val_dataset = VIDDataset(val_imdb, data_dir, config, valid_z_transforms, valid_x_transforms, "Validation")

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.train_num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.val_num_workers, drop_last=True)

    # create SiamFC network architecture (see details in SiamNet.py)
    net = SiamNet()
    # move network to GPU if using GPU
    if use_gpu:
        net.cuda()

    # define training strategy;
    # the learning rate of adjust layer (i.e., a conv layer)
    # is set to 0 as in the original paper
    optimizer = torch.optim.SGD([
        {'params': net.feat_extraction.parameters()},
        {'params': net.adjust.bias},
        {'params': net.adjust.weight, 'lr': 0},
    ], config.lr, config.momentum, config.weight_decay)

    # adjusting learning in each epoch
    scheduler = StepLR(optimizer, config.step_size, config.gamma)

    # used to control generating label for training;
    # once generated, they are fixed since the labels for each
    # pair of images (examplar z and search region x) are the same
    train_response_flag = False
    valid_response_flag = False

    # ------------------------ training & validation process ------------------------
    for i in range(config.num_epoch):

        # adjusting learning rate
        scheduler.step()

        # ------------------------------ training ------------------------------
        # indicating training (very importance for batch normalization)
        net.train()

        # used to collect loss
        train_loss = []

        for j, data in enumerate(tqdm(train_loader)):

            # fetch data, i.e., B x C x W x H (batchsize x channel x wdith x heigh)
            exemplar_imgs, instance_imgs = data

            # forward pass
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))

            # create label for training (only do it one time)
            if not train_response_flag:
                # change control flag
                train_response_flag = True
                # get shape of output (i.e., response map)
                response_size = output.shape[2:4]
                # generate label and weight
                train_eltwise_label, train_instance_weight = create_label(response_size, config, use_gpu)

            # clear the gradient
            optimizer.zero_grad()

            # loss
            loss = net.weight_loss(output, train_eltwise_label, train_instance_weight)

            # backward
            loss.backward()

            # update parameter
            optimizer.step()

            # collect training loss
            train_loss.append(loss.data)

        # ------------------------------ saving model ------------------------------
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(net, model_save_path + "SiamFC_" + str(i + 1) + "_model.pth")

        # ------------------------------ validation ------------------------------
        # indicate validation
        net.eval()

        # used to collect validation loss
        val_loss = []

        for j, data in enumerate(tqdm(val_loader)):

            exemplar_imgs, instance_imgs = data

            # forward pass
            output = net.forward(Variable(exemplar_imgs.cuda()), Variable(instance_imgs.cuda()))

            # create label for validation (only do it one time)
            if not valid_response_flag:
                valid_response_flag = True
                response_size = output.shape[2:4]
                valid_eltwise_label, valid_instance_weight = create_label(response_size, config, use_gpu)

            # loss
            loss = net.weight_loss(output, valid_eltwise_label, valid_instance_weight)

            # collect validation loss
            val_loss.append(loss.data)

        print ("Epoch %d   training loss: %f, validation loss: %f" % (i+1, np.mean(train_loss), np.mean(val_loss)))


if __name__ == "__main__":

    data_dir = "/home/hfan/Dataset/ILSVRC2015_crops/Data/VID/train"
    train_imdb = "/home/hfan/Desktop/PyTorch-SiamFC/ILSVRC15-curation/imdb_video_train.json"
    val_imdb = "/home/hfan/Desktop/PyTorch-SiamFC/ILSVRC15-curation/imdb_video_val.json"

    # training SiamFC network, using GPU by default
    train(data_dir, train_imdb, val_imdb)