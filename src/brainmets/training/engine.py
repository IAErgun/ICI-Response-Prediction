"""Training and evaluation loops.

Extracted from the original per-experiment training scripts with an identical parse
tree. `train` is the
version from the multiclass scripts, which is a strict superset of the binary one: the
same loop plus per-class confusion/recall logging (guarded by ``num_classes > 1``) and
two extra return values, which are None for binary runs.
"""
import datetime
import os
import torch

from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from monai.transforms import (Activations, AsDiscrete, Compose)
from . import compat  # noqa: F401 - patches torch.load on import
from .models import Custom3DCNN, use_cropped_images_fn


def per_class_metrics(y_true, y_pred, num_classes):
    """Confusion matrix (rows=true, cols=pred) and per-class recall over a fixed
    label set, so a class that's never predicted still shows up (recall 0)."""
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return cm, recall


def log_per_class(writer, split, y_true, y_pred, num_classes, epoch):
    cm, recall = per_class_metrics(y_true, y_pred, num_classes)
    for c in range(num_classes):
        writer.add_scalar(f"{split}/Recall_class{c}", recall[c], epoch)
    print(f"  {split} per-class recall: " +
          ", ".join(f"c{c}={recall[c]:.3f}" for c in range(num_classes)))
    print(f"  {split} confusion (rows=true, cols=pred):\n{cm}")
    return cm, recall


def train(train_loader, val_loader, num_classes, seq_of_interest, log_dir_backbone, model_name, device='cuda', learning_rate=0.005, wd=0.000025, max_epochs=100, cohen_kappa_score_weights='linear', key_metric_train_limit=0.68):
    in_channels = len(seq_of_interest)
    model = Custom3DCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    if num_classes > 1:
        loss_function = torch.nn.CrossEntropyLoss()
        post_pred = Compose([Activations(softmax=True, dim=1)])
        post_label = Compose([AsDiscrete(argmax=True, dim=1)])
    else:
        loss_function = torch.nn.functional.binary_cross_entropy_with_logits
        post_pred = Compose([Activations(sigmoid=True)])
        post_label = Compose([AsDiscrete(threshold=0.5)])
    # optimizer and scheduler
    optimizer = torch.optim.SGD(list(model.parameters()), lr=learning_rate, momentum=0.99, dampening=0, weight_decay=wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20,25,30,35,40,45,50,60,70,80,90,100], gamma=0.75)
    # Training parameters
    best_metric = -1
    best_metric_epoch = -1
    best_val_cm = None       # confusion matrix at the checkpoint-selected epoch
    best_val_recall = None   # per-class recall at the checkpoint-selected epoch
    log_filename = model_name.replace(".pth", "")
    log_dir = os.path.join(log_dir_backbone, log_filename)
    current_time = datetime.datetime.now().time()
    today = datetime.date.today()
    epoch_probabilities_class_0 = []
    epoch_probabilities_class_1 = []
    epoch_probabilities_class_0_val = []
    epoch_probabilities_class_1_val = []
    writer = SummaryWriter(
        log_dir=f"{log_dir}_{today}_{current_time}"
    )
    for epoch in range(max_epochs):
        print(f"epoch {epoch + 1}/{max_epochs}")
        model = model.train()
        epoch_loss = 0
        step = 0
        all_predicted = []
        all_labels = []
        all_predicted_val = []
        all_labels_val = []
        y_pred_train = torch.tensor([], dtype=torch.float32, device=device)
        y_train = torch.tensor([], dtype=torch.long, device=device)
        y_pred_train_prob = torch.tensor([], dtype=torch.float32, device=device)
        y_pred_val = torch.tensor([], dtype=torch.float32, device=device)
        y_val = torch.tensor([], dtype=torch.long, device=device)
        y_pred_val_prob = torch.tensor([], dtype=torch.float32, device=device)
        probs_class_0 = []
        probs_class_1 = []
        probs_class_0_val = []
        probs_class_1_val = []
        for batch_data in tqdm(train_loader):
            step += 1
            if in_channels == len(seq_of_interest):
                inputs = torch.cat(
                    [
                        batch_data[seq].to(device)
                        for seq in seq_of_interest
                    ],
                    dim=1,
                )
            else:
                print("Incorrect number of input channels")
            labels = batch_data["label"].to(device).float()
            optimizer.zero_grad()
            outputs = use_cropped_images_fn(inputs, batch_data, model, device=device, use_cropped_images=True)
            probabilities = post_pred(outputs)
            predicted = post_label(probabilities).squeeze(dim=1)
            y_train = torch.cat([y_train, labels], dim=0)
            y_pred_train = torch.cat([y_pred_train, predicted], dim=0)
            if num_classes == 1:  # y_pred_*_prob only feeds binary AUROC; skip for multiclass
                if probabilities.squeeze().shape == torch.Size([]):     # (squeeze() collapses a size-1
                    y_pred_train_prob = torch.cat([y_pred_train_prob, probabilities.squeeze().unsqueeze(0)], dim=0)
                else:                                                    # last batch [1,C]->[C] and breaks the cat)
                    y_pred_train_prob = torch.cat([y_pred_train_prob, probabilities.squeeze()], dim=0)
            if num_classes > 1:
                loss = loss_function(outputs, labels.to(torch.long))
            else:
                loss = loss_function(outputs, torch.unsqueeze(labels, axis=-1))
            loss.backward()
            optimizer.step()
            probabilities = probabilities.detach().cpu().numpy().flatten()
            for i, label in enumerate(labels):
                if label.item() == 0:
                    probs_class_0.append(
                        probabilities[i]
                    )
                else:
                    probs_class_1.append(probabilities[i])
            epoch_loss += loss.item()
            all_predicted.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        model = model.eval()
        step_val = 0
        for batch_data in tqdm(val_loader):
            step_val += 1
            if in_channels == len(seq_of_interest):
                inputs = torch.cat(
                    [
                        batch_data[seq].to(device)
                        for seq in seq_of_interest
                    ],
                    dim=1,
                )
            else:
                print("Incorrect number of input channels")
            labels = batch_data["label"].to(device).float()
            with torch.no_grad():
                outputs = use_cropped_images_fn(inputs, batch_data, model, device=device, use_cropped_images=True)
                probabilities = post_pred(outputs)
                predicted = post_label(probabilities).squeeze(dim=1)
                y_pred_val = torch.cat([y_pred_val, predicted], dim=0)
                y_val = torch.cat([y_val, labels], dim=0)
                if num_classes == 1:  # skip for multiclass (unused; squeeze breaks on size-1 last batch)
                    if probabilities.squeeze().shape == torch.Size([]):
                        y_pred_val_prob = torch.cat(
                            [y_pred_val_prob, probabilities.squeeze().unsqueeze(0)], dim=0
                        )
                    else:
                        y_pred_val_prob = torch.cat(
                            [y_pred_val_prob, probabilities.squeeze()], dim=0
                    )
                probabilities = probabilities.detach().cpu().numpy().flatten()
                for i, label in enumerate(labels):
                    if label.item() == 0:
                        probs_class_0_val.append(
                            probabilities[i]
                        )
                    else:
                        probs_class_1_val.append(probabilities[i])
                all_predicted_val.extend(predicted.detach().cpu().numpy())
                all_labels_val.extend(labels.detach().cpu().numpy())
        # Calculate average loss
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar("Train/Loss", epoch_loss, epoch + 1)
        epoch_probabilities_class_0.append(probs_class_0)
        epoch_probabilities_class_1.append(probs_class_1)
        train_kappa = cohen_kappa_score(all_labels, all_predicted, weights=cohen_kappa_score_weights)
        writer.add_scalar(f"Train/Cohen_Kappa", train_kappa, epoch + 1)
        print(
            f"Training Cohen Kappa: {train_kappa} for Epoch {epoch + 1}/{max_epochs}"
        )
        # MONAI AUC
        acc_value_train = torch.eq(y_pred_train, y_train)
        acc_metric_train = acc_value_train.sum().item() / len(y_train)
        if num_classes == 1:
            roc_scores_train = roc_auc_score(
                y_train.detach().cpu().numpy(), y_pred_train_prob.detach().cpu().numpy()
            )
            writer.add_scalar(f"Train/ROC_AUC", roc_scores_train, epoch + 1)
            print(
                f"Train ROC AUC: {roc_scores_train} for Epoch {epoch + 1}/{max_epochs}"
            )
        writer.add_scalar(f"Train/ACC", acc_metric_train, epoch + 1)
        print(
            f"Train ACC: {acc_metric_train} for Epoch {epoch + 1}/{max_epochs}"
        )
        # MONAI AUC VAL
        acc_value_val = torch.eq(y_pred_val, y_val)
        acc_metric_val = acc_value_val.sum().item() / len(y_val)
        if num_classes == 1:
            roc_scores_val = roc_auc_score(
                y_val.detach().cpu().numpy(), y_pred_val_prob.detach().cpu().numpy()
            )
            writer.add_scalar(f"Val/ROC_AUC", roc_scores_val, epoch + 1)
            print(
                f"Val ROC AUC: {roc_scores_val} for Epoch {epoch + 1}/{max_epochs}"
            )
        writer.add_scalar(f"Val/ACC", acc_metric_val, epoch + 1)
        print(
            f"Val ACC: {acc_metric_val} for Epoch {epoch + 1}/{max_epochs}"
        )
        val_kappa = cohen_kappa_score(all_labels_val, all_predicted_val, weights=cohen_kappa_score_weights)
        writer.add_scalar(f"Validation/Cohen_Kappa", val_kappa, epoch + 1)  #
        print(
            f"Validation Cohen Kappa: {val_kappa} for Epoch {epoch + 1}/{max_epochs}"
        )
        # Per-class recall + confusion matrix (multiclass only): overall ACC/kappa can
        # look fine while the minority class (Stable) is never predicted.
        if num_classes > 1:
            log_per_class(writer, "Train", all_labels, all_predicted, num_classes, epoch + 1)
            epoch_val_cm, epoch_val_recall = log_per_class(
                writer, "Val", all_labels_val, all_predicted_val, num_classes, epoch + 1)
        if num_classes == 1:
            key_metric_train = roc_scores_train
            key_metric_val = roc_scores_val
        else:
            key_metric_train = train_kappa
            key_metric_val = val_kappa
        if (key_metric_train > key_metric_train_limit) & (key_metric_val >= best_metric):
            best_metric_epoch = epoch + 1
            best_metric = key_metric_val
            torch.save(model.state_dict(), os.path.join(log_dir_backbone, model_name))
            if num_classes > 1:
                best_val_cm, best_val_recall = epoch_val_cm, epoch_val_recall
            print("saved new best metric model \n")
        scheduler.step()
    print(
        f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch} - - last metric value {key_metric_train:.4f}"
    )
    best_val_cm = best_val_cm.tolist() if best_val_cm is not None else None
    best_val_recall = best_val_recall.tolist() if best_val_recall is not None else None
    return key_metric_train, key_metric_val, best_metric, best_metric_epoch, best_val_cm, best_val_recall


def test(loader, model_save_path, num_classes, seq_of_interest, device='cuda', cohen_kappa_score_weights='linear', test_time_augmentation=[False, [2,3,4]]):
    if num_classes > 1:
        post_pred = Compose([Activations(softmax=True, dim=1)])
        post_label = Compose([AsDiscrete(argmax=True, dim=1)])
    else:
        post_pred = Compose([Activations(sigmoid=True)])
        post_label = Compose([AsDiscrete(threshold=0.5)])
    in_channels = len(seq_of_interest)
    model = Custom3DCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model = model.eval()
    all_predicted = []
    all_labels = []
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.long, device=device)
    y_pred_prob = torch.tensor([], dtype=torch.float32, device=device)
    probs_class_0 = []
    probs_class_1 = []
    if test_time_augmentation[0] == True:
        flip_orientations = [i for sublist in [list(combinations(test_time_augmentation[1], i)) for i in range(0, len(test_time_augmentation[1])+1)] for i in sublist]
    else:
        flip_orientations = [()]
    for batch_data in tqdm(loader):
        if in_channels == len(seq_of_interest):
            inputs = torch.cat(
                [
                    batch_data[seq].to(device)
                    for seq in seq_of_interest
                ],
                dim=1,
            )
        else:
            print("Incorrect number of input channels")
        labels = batch_data["label"].to(device).float()
        with torch.no_grad():
            outputs = torch.mean(torch.stack([torch.squeeze(use_cropped_images_fn(torch.flip(inputs, flip_orientation), batch_data, model, device=device, use_cropped_images=True)) for flip_orientation in flip_orientations]), axis=0)
            outputs = torch.unsqueeze(outputs, dim=1)
            probabilities = post_pred(outputs)
            predicted = post_label(probabilities).squeeze()
            y_pred = torch.cat([y_pred, predicted], dim=0)
            y = torch.cat([y, labels], dim=0)
            if probabilities.squeeze().shape == torch.Size([]):
                y_pred_prob = torch.cat(
                    [y_pred_prob, probabilities.squeeze().unsqueeze(0)], dim=0
                )
            else:
                y_pred_prob = torch.cat(
                    [y_pred_prob, probabilities.squeeze()], dim=0
                )
            probabilities = probabilities.detach().cpu().numpy().flatten()
            for i, label in enumerate(labels):
                if label.item() == 0:
                    probs_class_0.append(
                        probabilities[i]
                    )
                else:
                    probs_class_1.append(probabilities[i])
            all_predicted.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    # MONAI AUC VAL
    acc_value = torch.eq(y_pred, y)
    acc_metric = acc_value.sum().item() / len(y)
    if num_classes == 1:
        roc_scores = roc_auc_score(
            y.detach().cpu().numpy(), y_pred_prob.detach().cpu().numpy()
        )
        print(
            f"ROC AUC: {roc_scores}"
        )
    print(
        f"ACC: {acc_metric}"
    )
    kappa = cohen_kappa_score(all_labels, all_predicted, weights=cohen_kappa_score_weights)
    print(
        f"Cohen Kappa: {kappa}"
    )
    return y_pred.detach().cpu().numpy(), y_pred_prob.detach().cpu().numpy(), y.detach().cpu().numpy()
