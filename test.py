import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import transforms as T
from my_dataset import FS2K_DataSet_With_Seg
from models.swin import FaceAttrModel
from utils import read_train_test, checkpoint_load, evaluate

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_attrs, test_attrs = read_train_test()

    data_transform = {
        "train": T.Compose([
            T.RandomResize(224),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(0.5)]),

        "test": T.Compose([
            T.CenterCrop(224),
        ])}
    img_data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    val_dataset = FS2K_DataSet_With_Seg(attrs=test_attrs, transform=data_transform["test"], t2=img_data_transform)

    batch_size = args.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 6
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    model = FaceAttrModel().to(device)
    if args.weights != "":
        checkpoint_load(model, args.weights, device)

    # validate
    val_loss, val_accuracy_hair, val_accuracy_hair_color, val_accuracy_gender, val_accuracy_earring, val_accuracy_smile, \
    val_accuracy_frontal_face, val_accuracy_style = evaluate(model=model, data_loader=val_loader, device=device, epoch=0)
    val_mAP = (val_accuracy_hair.item() + val_accuracy_style.item() + val_accuracy_gender.item() + val_accuracy_earring.item() +
                val_accuracy_smile.item() + val_accuracy_frontal_face.item()) / 6

    print(
        "test: hair: {:.3f}, hair_color: {:.3f}, gender: {:.3f}, earring: {:.3f}, smile: {:.3f}, frontal_face: {:.3f}, style: {:.3f}, mAP: {:.3f}".format(
            val_accuracy_hair.item(), val_accuracy_hair_color.item(), val_accuracy_gender.item(), val_accuracy_earring.item(),
            val_accuracy_smile.item(), val_accuracy_frontal_face.item(), val_accuracy_style.item(), val_mAP))

    # 混淆矩阵
    # visualize_grid(model, val_loader, device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--weights', type=str, default='weights/model_best.pth',
                        help='initial weights path')

    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
