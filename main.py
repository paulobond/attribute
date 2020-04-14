from torchvision import models
import torch
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import pickle
from torch.utils.data import Dataset
from collections import defaultdict
from torch.optim import lr_scheduler
from attribute_index import attribute2attribute_index, n_attributes, attribute_index2attribute
from model import Model


use_cuda = torch.cuda.is_available()
print(f"Use cuda: {use_cuda}")

product_label2votes = pickle.load(open('./label2votes.pkl', 'rb'))

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root="./dataset/", transform=data_transforms)
dataset_idx2product_label = {v:k for k,v in dataset.class_to_idx.items()}

dataset_idx2y = {}

for dataset_idx, product_label in dataset_idx2product_label.items():
    y = torch.zeros(n_attributes)
    if product_label in product_label2votes:
        votes = product_label2votes[product_label]
        for att_category, dic2 in votes.items():
            for att_value, n_votes in dic2.items():
                if n_votes >= 2:
                    attribute_index = attribute2attribute_index[att_category][att_value]
                    y[attribute_index] = 1.
        dataset_idx2y[dataset_idx] = y

attribute_index2n_product_labels = defaultdict(int)

for _, y in dataset_idx2y.items():
    for i in range(len(y)):
        if y[i] == 1.0:
            attribute_index2n_product_labels[i] += 1


model = Model()
if use_cuda:
    model.cuda()

train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                           [len(dataset)-1000, 1000])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=10,
                                           shuffle=True,
                                           drop_last=False,
                                           num_workers=4)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=10,
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=4)

optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                      lr=0.1, momentum=0.5)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for epoch in range(1, 10):

    if epoch > 10:
        model.unfreeze()
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),
                              lr=exp_lr_scheduler.get_lr()[0],
                              momentum=0.5)

    print(f"Train epoch {epoch}. Learning rate: {exp_lr_scheduler.get_lr()[0]}")
    # train
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}")

        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        target2 = torch.zeros(len(target), n_attributes)
        for i in range(len(target2)):
            target2[i] = dataset_idx2y[int(target[i])]
        if use_cuda:
            target2 = target2.cuda()

        criterion = torch.nn.BCELoss(reduction='mean')
        loss = criterion(output, target2)
        train_loss += criterion(output, target2).data.item()
        loss.backward()
        optimizer.step()

    exp_lr_scheduler.step()

    train_loss /= len(train_loader.dataset)
    print("------" * 10)
    print(f"Train loss: {train_loss}")

    # eval
    print(f"Validation epoch {epoch}")
    model.eval()
    validation_loss = 0
    exact_matches = 0
    hamming_score = 0

    example_based_recall = 0
    example_based_precision = 0

    label_based_TP = torch.zeros(n_attributes)
    label_based_TN = torch.zeros(n_attributes)
    label_based_FP = torch.zeros(n_attributes)
    label_based_FN = torch.zeros(n_attributes)

    for batch_idx, (data, target) in enumerate(val_loader):

        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        predictions = (output > 0.5).type(torch.int)

        target2 = torch.zeros(len(target), n_attributes)
        for i in range(len(target2)):
            target2[i] = dataset_idx2y[int(target[i])]
        if use_cuda:
            target2 = target2.cuda()

        # validation loss
        criterion = torch.nn.BCELoss(reduction='mean')
        validation_loss += criterion(output, target2).data.item()

        # exact matches
        for i in range(len(predictions)):
            if all(predictions[i] == target2[i]):
                exact_matches += 1

        # hamming loss
        for i in range(len(predictions)):
            hamming_score += int(sum(target2[i] == predictions[i])) / len(target2[i])

        # Example based recall
        for i in range(len(predictions)):
            example_based_recall += int(sum((target2[i] == 1) & (predictions[i] == 1))) / int(sum(target2[i] == 1))

        # Example based precision
        for i in range(len(predictions)):
            example_based_precision += int(sum((target2[i] == 1) & (predictions[i] == 1))) / int(
                sum(predictions[i] == 1))

        # Label based metrics
        label_based_TP += ((predictions == 1) & (target2 == 1)).sum(dim=0)
        label_based_TN += ((predictions == 0) & (target2 == 0)).sum(dim=0)
        label_based_FP += ((predictions == 1) & (target2 == 0)).sum(dim=0)
        label_based_FN += ((predictions == 0) & (target2 == 1)).sum(dim=0)

    validation_loss /= len(val_loader.dataset)
    exact_matches /= len(val_loader.dataset)
    hamming_score /= len(val_loader.dataset)
    example_based_recall /= len(val_loader.dataset)
    example_based_precision /= len(val_loader.dataset)

    print(f"Epoch {epoch}")
    print(f"Validation loss: {validation_loss}")
    print(f"Exact match proportion: {exact_matches}")
    print(f"Hamming score / global accuracy : {hamming_score}")
    print(f"Mean example recall: {example_based_recall}")
    print(f"Mean example precision: {example_based_precision}")
    for attribute_index in range(n_attributes):
        if float(label_based_TP[attribute_index] + label_based_FN[attribute_index]) <= 0:
            mean_recall = "NA"
        else:
            mean_recall = round(float(
                label_based_TP[attribute_index] / (label_based_TP[attribute_index] + label_based_FN[attribute_index])),
                                2)

        if float(label_based_TP[attribute_index] + label_based_FP[attribute_index]) <= 0:
            mean_precision = "NA"
        else:
            mean_precision = round(float(
                label_based_TP[attribute_index] / (label_based_TP[attribute_index] + label_based_FP[attribute_index])),
                                   2)
        print(f"Attribute {attribute_index} ({attribute_index2attribute[attribute_index]}):"
              f" precision {mean_precision}  recall {mean_recall}")
