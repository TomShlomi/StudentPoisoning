import argparse
import math
import PIL
import pickle
import torch
import torchvision
import multiprocessing as mp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import SimpleCNN, MediumCNN
from poison_data import poison_images_with_tom_patch_dataset
from resnet import resnet18, resnet50
from tests import clean_accuracy, clean_accuracy_per_class, trigger_prob_increase, non_target_trigger_success

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')


def find_optimal_num_workers(dataset, batch_size):
        
        best_workers, min_time = None, math.inf
        print(f"Batch size {batch_size}")
        print(f"Num workers range: {2} to {mp.cpu_count()}")
        
        for num_workers in range(2, mp.cpu_count(), 2):
            print(f"Testing with {num_workers}")
            train_loader = DataLoader(dataset,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      batch_size=batch_size,
                                      pin_memory=True
                                      )
            # Track iteration time
            start = time()
            for _ in train_loader:
                pass
            end = time()
            
            print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

            if end - start < min_time:
                best_workers, min_time = num_workers, end - start

        print("Done testing for optimal num_workers")
        return best_workers


def create_poisoned_data(batch_size, teacher, raw_train_set, new_patch=False, new_train_set=False, perturb=False):

    save_file = 'perturbed_poisoned_trainset_no_label.pkl' if perturb else 'poisoned_trainset.pkl'
    print(f"Creating poisoned data for {save_file}")
    
    # Poison dataset
    if new_patch:
        # Create random patch that the teacher learns to associate with the first class
        patch = torch.randint(0, 2, (4, 4)).to(torch.float32)
        patch = torch.stack((patch, patch, patch), 0)
        patchim = transforms.ToPILImage()(patch)
        patchim.save('patch.png')
    else:
        patch = transforms.ToTensor()(PIL.Image.open('patch.png'))

    patch = patch.to(device)

    if new_train_set:
        poisoned_trainset = poison_images_with_tom_patch_dataset(
            teacher=teacher, 
            raw_train_set=raw_train_set, 
            patch=patch, 
            steps=2, 
            threshold=1, 
            perturb=perturb, 
            verbose=True, 
            epsilon=0.05
        )
        with open(save_file, 'wb') as f:
            pickle.dump(poisoned_trainset, f)
    else:
        with open(save_file, 'rb') as f:
            poisoned_trainset = pickle.load(f)

    # print("poisoned_trainset[0][0]", poisoned_trainset[0][0].shape)
    # print("poisoned_trainset[0][1]", poisoned_trainset[0][1].shape)
    # print("poisoned_trainset[0][2]", poisoned_trainset[0][2].shape)

    poison_loader = DataLoader(poisoned_trainset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    return poisoned_trainset, poison_loader, patch


def mix_with_clean_dataset(args, teacher, normalized_trainset, poisoned_trainset, poisoned_percentage=0.01, new_mix_probs=False):
    """
    Mix clean and poisoned dataset.
    More clean examples allow model to learn robust features.
    More poisoned examples push model to learn non-robust patch.
    """
    print(f"Mixing datasets with {poisoned_percentage} proportion of poisoned images")
    clean_file = 'clean_trainset.pkl'
    # clean_file = 'clean_trainset_resnet_18.pkl'

    if new_mix_probs:
        print("Creating new Clean Trainset")
        clean_trainset = []
        
        with torch.no_grad():
            teacher.eval()
            for i, (image, label) in enumerate(normalized_trainset):

                if i % 500 == 0:
                    print(f"Generated {i} clean datapoints")

                image = image.to(device)
                probs = teacher(image.reshape((1, 3, 32, 32))).softmax(dim=-1)
                clean_trainset.append((image.cpu(), probs.cpu(), label))

        with open(clean_file, 'wb') as f:
            pickle.dump(clean_trainset, f)

    else:
        with open(clean_file, 'rb') as f:
            clean_trainset = pickle.load(f)

    # print("clean_trainset[0][0]:", clean_trainset[0][0].shape)
    # print("clean_trainset[0][1]:", clean_trainset[0][1].shape)
    # print("clean_trainset[0][2]:", clean_trainset[0][2].shape)

    num_poison = len(poisoned_trainset)
    poisoned_indices = np.random.choice(num_poison, int(num_poison * poisoned_percentage), replace=False)
    poisoned_indices = np.array(poisoned_indices)

    clean_indices = np.array(list(
        set(range(num_poison)) - set(poisoned_indices)
    ))
    
    mixed_trainset = [poisoned_trainset[i] for i in poisoned_indices] + [clean_trainset[i] for i in clean_indices]
    return mixed_trainset


def loss_fn_kd(outputs, labels, teacher_outputs, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities!
    """
    
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
        F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, choices=["cifar10", "imagenet"], default="cifar10")
    parser.add_argument("--new-patch", "-np", action="store_true")
    parser.add_argument("--new-dataset", "-nd", action="store_true")
    parser.add_argument("--new-clean-dataset", "-ncd", action="store_true")
    parser.add_argument("--perturb-raw", "-pr", action="store_true")
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--teacher-model", "-tm", type=str, default="medium-cnn")
    parser.add_argument("--student-model", "-sm", type=str, default="medium-cnn")
    parser.add_argument("--num-workers", "-nw", type=int, default=8)
    parser.add_argument("--find-optimal-workers", "-fow", action="store_true")
    parser.add_argument("--mix-data", "-m", action="store_true")
    parser.add_argument("--poison-percentage", "-pp", type=float, default=0.1)
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--target-label", "-tl", type=int, default=0)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
    parser.add_argument("--temperature", "-tmp", type=float, default=5)
    parser.add_argument("--alpha", "-a", type=float, default=0.9)
    args = parser.parse_args()

    writer = SummaryWriter()

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    # Load CIFAR-10
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformations)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transformations)
    
    # Need raw datasets for poisoning
    raw_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    raw_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    # Use Tensor datasets with no poisoning to evaluate accuracy on full set of clean images
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Load trained teacher
    if args.teacher_model == "simple-cnn":
        teacher = SimpleCNN(c_in=3, w_in=32, h_in=32, num_classes=10)
        teacher.load_state_dict(torch.load('small-teacher.pt'))
    elif args.teacher_model == "medium-cnn":
        teacher = MediumCNN(c_in=3, w_in=32, h_in=32, num_classes=10)
        teacher.load_state_dict(torch.load('medium-teacher.pt'))
    elif args.teacher_model == "resnet-18":
        teacher = resnet18()
        teacher.load_state_dict(torch.load('resnet18-cifar.pt'))
    elif args.teacher_model == "resnet-50":
        teacher = resnet50()
        teacher.load_state_dict(torch.load('resnet50-cifar.pt'))

    teacher.to(device)

    poisoned_trainset, poison_loader, patch = create_poisoned_data(
        args.batch_size, 
        teacher, 
        raw_train_set, 
        new_patch=args.new_patch, 
        new_train_set=args.new_dataset, 
        perturb=args.perturb_raw,
    )
    
    if args.mix_data:
        poisoned_trainset = mix_with_clean_dataset(args, teacher, train_set, poisoned_trainset, poisoned_percentage=args.poison_percentage, new_mix_probs=args.new_clean_dataset)

    if args.find_optimal_workers:
        find_optimal_num_workers(poisoned_trainset, args.batch_size)

    poison_loader = DataLoader(poisoned_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Specify student model
    if args.student_model == "simple-cnn":
        student = SimpleCNN(c_in=3, w_in=32, h_in=32, num_classes=10)
    elif args.student_model == "medium-cnn":
        student = MediumCNN(c_in=3, w_in=32, h_in=32, num_classes=10)
    elif args.student_model == "resnet-18":
        student = resnet18()

    student.to(device)
    
    learning_rate = args.learning_rate
    optimizer = torch.optim.SGD(student.parameters(), learning_rate)

    clean_acc = clean_accuracy(teacher, test_loader)
    print(f'TEACHER SANITY CHECK -- Clean Accuracy is {clean_acc:.3f}')

    for i in range(args.epochs):
        student.train()
        teacher.eval()
        for j, data in enumerate(poison_loader):
            
            images, probs, labels = data
            images, probs, labels = images.to(device), probs.to(device), labels.to(device)
            
            optimizer.zero_grad()
        
            student_outputs = student(images)
            teacher_outputs = teacher(images)
            KD_loss = loss_fn_kd(student_outputs, labels, teacher_outputs, alpha=args.alpha, T=args.temperature)
            KD_loss.backward()

            optimizer.step()
            
            if j % 100 == 0:
                print('Epoch: %d, Batch: %d, Loss: %.4f' % (i, j, KD_loss.item()))
        
        if i % 20 == 0:
            bleh = clean_accuracy_per_class(student, test_loader, classes)
            print('Accuracy per class:', bleh)
            print('Clean Accuracy:', sum(bleh) / len(bleh))
            prob_increase = trigger_prob_increase(student, test_set, patch, n=100)
            print(f'Trigger Target Probability Increase after {i + 1} epochs: {prob_increase:.3f}')
            non_target = non_target_trigger_success(model=student, clean_dataset=test_set, raw_dataset=raw_test_set, patch=patch, target=args.target_label)
            print(f'Non-Target Trigger Success after {i + 1} epochs: {non_target:.3f}')
        
        torch.save(student.state_dict(), 'actually-smart-student-fuck.pt')

        print("\n")

    # Test student
    print(clean_accuracy(student, test_loader)_
    print('Accuracy per class:', clean_accuracy_per_class(student, test_loader, classes))
    print('Non-Target Trigger Success after:', non_target_trigger_success(model=student, clean_dataset=test_set, raw_dataset=raw_test_set, patch=patch, target=args.target_label))

    bleh = clean_accuracy_per_class(student, test_loader, classes)
    print('Accuracy per class:', bleh)
    print('Clean Accuracy:', sum(bleh) / len(bleh))