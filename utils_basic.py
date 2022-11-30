from enum import Enum
from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import matplotlib.pyplot as plt


def train_model(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    num_epochs: int,
    num_classes: int,
    eval_interval: int,
    is_binary: bool,
    gpu=True,
    writer: Optional[SummaryWriter] = None,
):
    """
    A standard training loop for the basic model.
    :param eval_interval: evaluate every `eval_interval` epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    if writer is None:
        close_writer = True
        writer = SummaryWriter()
    else:
        close_writer = False

    t = trange(num_epochs, desc="epochs")
    show_batches = len(trainloader) >= 480  # magic number
    step = 0
    for epoch in t:
        model.train()  # reset each epoch since evaluation sets it to eval
        cum_loss = 0.0
        cum_acc = 0.0
        tot = 0.0

        batches = tqdm(enumerate(trainloader)) if show_batches else enumerate(trainloader)
        for x_in, y_in in batches:
            B = x_in.size(0)  # the batch size
            if gpu:
                x_in, y_in = x_in.cuda(), y_in.cuda()
            pred = model(x_in)
            loss = model.loss(pred, y_in)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss.item() * B
            if is_binary:
                acc = ((pred > 0).long().eq(y_in)).sum().item()
            else:
                pred_c = pred.max(1)[1]
                acc = (pred_c.eq(y_in)).sum().item()
            tot = tot + B
            cum_acc += acc

            writer.add_scalar("train/loss", loss.item(), global_step=step)
            writer.add_scalar("train/acc", acc, global_step=step)
            step += 1

        if epoch % eval_interval == eval_interval - 1 or epoch == num_epochs - 1:
            t.set_description(f"[Epoch {epoch}] Evaluating...")
            (
                acc,
                acc_poison,
                trigger_effect,
                class_acc,
                class_acc_poison,
                class_trigger_effect,
                class_totals,
            ) = eval_model(
                model,
                testloader,
                is_binary=is_binary,
                gpu=gpu,
                num_classes=num_classes,
            )

            for label, avg, classes in (
                ("acc_benign", acc, class_acc),
                ("acc_poisoned", acc_poison, class_acc_poison),
                ("trigger_effectiveness", trigger_effect, class_trigger_effect),
            ):
                writer.add_scalar("eval/acc/%s" % label, avg, epoch)
                writer.add_scalars(
                    "eval/acc/%s/classes" % label,
                    {str(i): classes[i] / class_totals[i] for i in range(num_classes)},
                    epoch,
                )

        t.set_description(
            "[Epoch %d] loss = %.4f, acc = %.4f"
            % (epoch, cum_loss / tot, cum_acc / tot)
        )

        scheduler.step()

    if close_writer:
        writer.close()

    # return the accuracy on the benign and poisoned datasets after last epoch
    return acc, acc_poison, trigger_effect


@torch.no_grad()
def eval_model(
    model: nn.Module,
    dataloader: DataLoader[Tuple[Tensor, Tensor, int, int]],
    is_binary: bool,
    num_classes: int,
    gpu=True,
):
    """
    A typical evaluation loop.
    :param dataloader: a dataloader that returns the original image, patched image, original label, and true label.
    :return:
    - the average accuracy across benign images
    - the avg acc across poisoned images
    - the percentage of images where the trigger caused the model to misclassify
    - the base counts of the above, plus the number of images per class
    """
    model.eval()

    # fast construction of some dummy vectors
    cum_acc_benign, cum_acc_poison, num_swapped, class_totals = torch.zeros(
        (4, num_classes)
    ).long()

    for X, X_patch, y, y_patch in dataloader:
        B = X.size(0)
        if gpu:
            X, X_patch = X.cuda(), X_patch.cuda()
        pred_benign, pred_poison = model(torch.cat((X, X_patch))).split(B)
        if is_binary:
            pred_benign, pred_poison = (pred_benign > 0).long(), (
                pred_poison > 0
            ).long()
        else:
            pred_benign, pred_poison = pred_benign.argmax(dim=-1), pred_poison.argmax(
                dim=-1
            )
        pred_benign, pred_poison = pred_benign.cpu(), pred_poison.cpu()

        cum_acc_benign.scatter_add_(0, y, (pred_benign == y).long())
        cum_acc_poison.scatter_add_(0, y, (pred_poison == y).long())
        num_swapped.scatter_add_(
            0, y, ((pred_poison == y_patch) & (pred_benign != y_patch)).long()
        )
        class_totals.scatter_add_(0, y, torch.ones(B, dtype=torch.long))

    return (
        cum_acc_benign.sum() / class_totals.sum(),
        cum_acc_poison.sum() / class_totals.sum(),
        num_swapped.sum() / class_totals.sum(),
        cum_acc_benign,
        cum_acc_poison,
        num_swapped,
        class_totals,
    )


def plt_imshow(img: Tensor, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = (img / 2 + 0.5).numpy()
    if one_channel:
        plt.imshow(img, cmap="greys")
    else:
        plt.imshow(np.transpose(img, (1, 2, 0)))
