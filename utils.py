import os
import sys
import json
import pickle
import random
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, accuracy_score
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from my_dataset import mean, std

import matplotlib.pyplot as plt
import shutil


def add_suffix(file_path):
    if os.path.exists(file_path + '.jpg'):
        file_path += '.jpg'
    else:
        file_path += '.png'
    return file_path


def read_train_test():
    json_files = {
        'train': '/data/yifan/FS2K/anno_train.json',
        'test': '/data/yifan/FS2K/anno_test.json'
    }

    for data_split, json_file in json_files.items():
        with open(json_file, 'r') as f:
            json_data = json.loads(f.read())

        attrs = {}
        for attr in json_data[0].keys():
            attrs[attr] = []

        for idx_fs, fs in enumerate(json_data):
            for attr in fs:
                if attr == 'image_name':
                    style, _path = fs[attr].split('/')
                    img_path = add_suffix(os.path.join('/data/yifan/FS2K', data_split, 'sketch_raw',
                                                       style[-1] + _path.replace('image', '_image')))
                    attrs[attr].append(img_path)
                else:
                    attrs[attr].append(fs[attr])

        if data_split == 'train':
            train_attrs = attrs
        else:
            test_attrs = attrs

    return train_attrs, test_attrs


def calculate_metrics(output, target):
    _, predicted_hair = output['hair'].max(1)
    gt_hair = target['hair']
    _, predicted_hair_color = output['hair_color'].max(1)
    gt_hair_color = target['hair_color']
    _, predicted_gender = output['gender'].max(1)
    gt_gender = target['gender']
    _, predicted_earring = output['earring'].max(1)
    gt_earring = target['earring']
    _, predicted_smile = output['smile'].max(1)
    gt_smile = target['smile']
    _, predicted_frontal_face = output['frontal_face'].max(1)
    gt_frontal_face = target['frontal_face']
    _, predicted_style = output['style'].max(1)
    gt_style = target['style']

    with warnings.catch_warnings():  # sklearn 在处理混淆矩阵中的零行时可能会产生警告
        warnings.simplefilter("ignore")
        accuracy_hair = accuracy_score(y_true=gt_hair.cpu().numpy(), y_pred=predicted_hair.cpu().numpy())
        accuracy_hair_color = accuracy_score(y_true=gt_hair_color.cpu().numpy(), y_pred=predicted_hair_color.cpu().numpy())
        accuracy_gender = accuracy_score(y_true=gt_gender.cpu().numpy(), y_pred=predicted_gender.cpu().numpy())
        accuracy_earring = accuracy_score(y_true=gt_earring.cpu().numpy(), y_pred=predicted_earring.cpu().numpy())
        accuracy_smile = accuracy_score(y_true=gt_smile.cpu().numpy(), y_pred=predicted_smile.cpu().numpy())
        accuracy_frontal_face = accuracy_score(y_true=gt_frontal_face.cpu().numpy(), y_pred=predicted_frontal_face.cpu().numpy())
        accuracy_style = accuracy_score(y_true=gt_style.cpu().numpy(), y_pred=predicted_style.cpu().numpy())
    return accuracy_hair, accuracy_hair_color, accuracy_gender, accuracy_earring, accuracy_smile, accuracy_frontal_face, accuracy_style


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    accuracy_hair = torch.zeros(1).to(device)
    accuracy_hair_color = torch.zeros(1).to(device)
    accuracy_gender = torch.zeros(1).to(device)
    accuracy_earring = torch.zeros(1).to(device)
    accuracy_smile = torch.zeros(1).to(device)
    accuracy_frontal_face = torch.zeros(1).to(device)
    accuracy_style = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data['img'], data['labels']
        sample_num += images.shape[0]

        pred = model(images.to(device), labels, device)

        loss_train, losses_train = model.get_loss(pred, labels, device)
        accu_loss += loss_train.item()
        batch_accuracy_hair, batch_accuracy_hair_color, batch_accuracy_gender, batch_accuracy_earring, \
        batch_accuracy_smile, batch_accuracy_frontal_face, batch_accuracy_style = calculate_metrics(pred, labels)

        accuracy_hair += batch_accuracy_hair
        accuracy_hair_color += batch_accuracy_hair_color
        accuracy_gender += batch_accuracy_gender
        accuracy_earring += batch_accuracy_earring
        accuracy_smile += batch_accuracy_smile
        accuracy_frontal_face += batch_accuracy_frontal_face
        accuracy_style += batch_accuracy_style

        loss_train.backward()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch+1, accu_loss.item() / (step + 1))

        if not torch.isfinite(loss_train):
            print('WARNING: non-finite loss, ending training ', loss_train)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accuracy_hair / (step + 1), accuracy_hair_color / (step + 1), accuracy_gender / (step + 1), \
           accuracy_earring / (step + 1), accuracy_smile / (step + 1), accuracy_frontal_face / (step + 1), accuracy_style / (step + 1)


def checkpoint_load(model, name, device):
    assert os.path.exists(name), "weights file: '{}' not exist.".format(name)
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location=device), strict=True)


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accuracy_hair = torch.zeros(1).to(device)
    accuracy_hair_color = torch.zeros(1).to(device)
    accuracy_gender = torch.zeros(1).to(device)
    accuracy_earring = torch.zeros(1).to(device)
    accuracy_smile = torch.zeros(1).to(device)
    accuracy_frontal_face = torch.zeros(1).to(device)
    accuracy_style = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data['img'], data['labels']
        sample_num += images.shape[0]

        pred = model(images.to(device), labels, device)
        # pred = model(images.to(device))
        val_loss, val_losses = model.get_loss(pred, labels, device)
        accu_loss += val_loss.item()

        batch_accuracy_hair, batch_accuracy_hair_color, batch_accuracy_gender, batch_accuracy_earring, \
        batch_accuracy_smile, batch_accuracy_frontal_face, batch_accuracy_style = calculate_metrics(pred, labels)
        accuracy_hair += batch_accuracy_hair
        accuracy_hair_color += batch_accuracy_hair_color
        accuracy_gender += batch_accuracy_gender
        accuracy_earring += batch_accuracy_earring
        accuracy_smile += batch_accuracy_smile
        accuracy_frontal_face += batch_accuracy_frontal_face
        accuracy_style += batch_accuracy_style

        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch+1, accu_loss.item() / (step + 1))

    return accu_loss.item() / (step + 1), accuracy_hair / (step + 1), accuracy_hair_color / (step + 1), accuracy_gender / (step + 1), \
           accuracy_earring / (step + 1), accuracy_smile / (step + 1), accuracy_frontal_face / (step + 1), accuracy_style / (step + 1)


