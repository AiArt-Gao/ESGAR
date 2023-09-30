import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import transforms as T
from torch.optim.lr_scheduler import LambdaLR

from my_dataset import  FS2K_DataSet_With_Seg
from models.swin import FaceAttrModel
from utils import read_train_test, train_one_epoch, evaluate, checkpoint_load


def linear_warmup_cosine_lr_scheduler(optimizer, warmup_time_ratio, T_max):
    T_warmup = int(T_max * warmup_time_ratio)

    def lr_lambda(epoch):
        # linear warm up
        if epoch < T_warmup:
            return epoch / T_warmup
        else:
            progress_0_1 = (epoch - T_warmup) / (T_max - T_warmup)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress_0_1))
            return cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_attrs, test_attrs = read_train_test()

    # 实例化训练数据集
    data_transform = {
        "train": T.Compose([
            T.RandomResize(224),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(0.5)]),

        "test": T.Compose([
            T.CenterCrop(224),
        ])}
    img_data_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = FS2K_DataSet_With_Seg(attrs=train_attrs, transform=data_transform["train"], t2=img_data_transform)
    val_dataset = FS2K_DataSet_With_Seg(attrs=test_attrs, transform=data_transform["test"], t2=img_data_transform)

    batch_size = args.batch_size

    nw = 6  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = FaceAttrModel().to(device)

    if args.weights != "":
        checkpoint_load(model, args.weights, device)

    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_map = 0
    for epoch in range(args.epochs):
        # train
        train_loss, accuracy_hair, accuracy_hair_color, accuracy_gender, accuracy_earring, \
        accuracy_smile, accuracy_frontal_face, accuracy_style = train_one_epoch(model=model, optimizer=optimizer,
                                                                                data_loader=train_loader, device=device,
                                                                                epoch=epoch)
        # 平均准确率
        train_mAP = (accuracy_hair.item() + accuracy_gender.item() + accuracy_earring.item()
                     + accuracy_smile.item() + accuracy_frontal_face.item() + accuracy_style.item()) / 6
        scheduler.step()

        # validate
        val_loss, val_accuracy_hair, val_accuracy_hair_color, val_accuracy_gender, \
        val_accuracy_earring, val_accuracy_smile, val_accuracy_frontal_face, val_accuracy_style \
            = evaluate(model=model, data_loader=val_loader, device=device, epoch=epoch)
        val_mAP1 = (
                          val_accuracy_hair.item() + val_accuracy_gender.item() + val_accuracy_earring.item() +
                          val_accuracy_smile.item() + val_accuracy_frontal_face.item() + val_accuracy_style.item()) / 6
        val_mAP = (
                          val_accuracy_hair.item() + val_accuracy_gender.item() +
                          val_accuracy_smile.item() + val_accuracy_frontal_face.item() + val_accuracy_style.item()) / 5

        print(
            "train: hair: {:.3f}, hair_color: {:.3f}, gender: {:.3f}, earring: {:.3f}, smile: {:.3f}, frontal_face: {:.3f}, style: {:.3f},train_map:{:.3f}".format(
                accuracy_hair.item(), accuracy_hair_color.item(), accuracy_gender.item(), accuracy_earring.item(),
                accuracy_smile.item(), accuracy_frontal_face.item(), accuracy_style.item(), train_mAP))
        print(
            "test: hair: {:.3f}, hair_color: {:.3f}, gender: {:.3f}, earring: {:.3f}, smile: {:.3f}, frontal_face: {:.3f}, style: {:.3f},val_map1:{:.3f} val_map:{:.3f}".format(
                val_accuracy_hair.item(), val_accuracy_hair_color.item(), val_accuracy_gender.item(),
                val_accuracy_earring.item(),
                val_accuracy_smile.item(), val_accuracy_frontal_face.item(), val_accuracy_style.item(), val_mAP1, val_mAP))

        if val_mAP > best_map:
            best_map = val_mAP
            torch.save(model.state_dict(), "./weights/model_best.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)

    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 2.5e-4)')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                                             help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
