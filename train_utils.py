import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ReduceLROnPlateau
import sklearn.metrics as metrics

from tqdm import tqdm

def validate(model, valloader, train_args):

    device = train_args["device"]
    model.to(device)
    model.eval()

    criterion = nn.BCELoss()
    running_loss = 0.0

    all_predictions = torch.zeros(len(valloader.dataset))
    all_labels = torch.zeros(len(valloader.dataset))

    for i, (input, label) in enumerate(valloader, 0):
        
        if i == 0:
            start = i*input.shape[0]
        else:
            start = end
        end = start + input.shape[0]

        input, label = input.float().to(device), label.float().to(device)
        with torch.no_grad():
            outputs = model(input)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, label)
            running_loss += loss

            all_predictions[start:end] = outputs.to("cpu").squeeze()
            all_labels[start:end] = label.to("cpu").squeeze()
    
    all_predictions = (all_predictions>0.5).long()
    all_labels = all_labels.long()
    
    mce_loss = running_loss/len(valloader.dataset)
    roc_auc_score = metrics.roc_auc_score(all_labels, all_predictions)
    accuracy = metrics.accuracy_score(all_labels, all_predictions)
    precision = metrics.precision_score(all_labels, all_predictions)
    recall = metrics.recall_score(all_labels, all_predictions)
    f1_score = metrics.f1_score(all_labels, all_predictions)

    return mce_loss, roc_auc_score, accuracy,  precision, recall, f1_score



def train(model, trainloader, valloader, train_args, take_last_model = False):
    
    device = train_args["device"]
    model.to(device)

    optimizer = Adam(model.parameters(), **train_args["optimizer_args"])
    has_scheduler = False
    if train_args["scheduler"] == "LinearLR":
        scheduler = LinearLR(optimizer=optimizer,total_iters=train_args["epochs"] ,verbose = False, start_factor= 0.5)
        has_scheduler = True
    if train_args["scheduler"] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=train_args["epochs"], eta_min=1e-8)
        has_scheduler = True
    if train_args["scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience = 5)
        has_scheduler = True
    criterion = nn.BCELoss()

    best_mce = 10e8
    best_roc_auc = 0
    result_dict = {}

    for epoch in tqdm(range(train_args["epochs"])):
        model.train()
        running_loss = 0.0
        all_predictions = torch.zeros(len(trainloader.dataset))
        all_labels = torch.zeros(len(trainloader.dataset))
        end = 0
        for i, (input, label) in enumerate(trainloader):

            input, label = input.float().to(device), label.float().to(device)

            optimizer.zero_grad()
            outputs = model(input)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            start = end
            end = start + input.shape[0]
            all_predictions[start:end] = outputs.detach().to("cpu").squeeze()
            all_labels[start:end] = label.to("cpu").squeeze()

            running_loss += loss.item()
        
        all_predictions = (all_predictions>0.5).long()
        all_labels = all_labels.long()
        train_mce = running_loss/len(trainloader.dataset)
        train_roc_auc_score = metrics.roc_auc_score(all_labels, all_predictions)
        train_accuracy = metrics.accuracy_score(all_labels, all_predictions)
        train_precision = metrics.precision_score(all_labels, all_predictions)
        train_recall = metrics.recall_score(all_labels, all_predictions)
        train_f1_score = metrics.f1_score(all_labels, all_predictions)
        val_mce_loss, val_roc_auc_score, val_accuracy,  val_precision, val_recall, val_f1_score = validate(model, valloader,train_args)

        if take_last_model or (val_mce_loss <= best_mce and val_roc_auc_score >= best_roc_auc):
            best_mce = val_mce_loss
            best_mce_model = copy.deepcopy(model)
            best_epoch = epoch
            best_roc_auc = val_roc_auc_score
        
        print(f"Epoch {epoch} | \t train loss:  {train_mce:.4f}, train roc auc: {train_roc_auc_score:.4f}, train accuracy: {train_accuracy:.4f}, train precision: {train_precision:.4f}, train recall: {train_recall:.4f}, train f1 score: {train_f1_score:.4f} | \
              \n\t val loss: {val_mce_loss:.4f}, val roc auc: {val_roc_auc_score:.4f}, val accuracy: {val_accuracy:.4f}, val precision: {val_precision:.4f}, val recall: {val_recall:.4f}, val f1 score: {val_f1_score:.4f} ", flush=True)
        
        if has_scheduler and train_args["scheduler"] != "ReduceLROnPlateau":
            scheduler.step()
        elif has_scheduler:
            scheduler.step(val_roc_auc_score)

    result_dict["mce"] = best_mce
    result_dict["best_epoch"] = best_epoch
    result_dict["model"] = copy.deepcopy(best_mce_model)
    result_dict["roc_auc"] = best_roc_auc
    print("Best model", best_epoch, best_mce, best_roc_auc)

    return result_dict