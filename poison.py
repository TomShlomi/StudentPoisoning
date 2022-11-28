import PIL
import pickle
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from modelsdefinitions import SimpleCNN, MediumCNN
from tests import testAccuracy, testAccuracyByClass, testPoisonSuccess, testPoisonSuccessPercent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: Fix these normalization params
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batchsize = 256
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transformations)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transformations)
rawtrainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transforms.ToTensor())
rawtestset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(
    trainset, batch_size=batchsize, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=batchsize,
                         shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Load trained teacher
teacher = MediumCNN(c_in=3, w_in=32, h_in=32, num_classes=10)
teacher.load_state_dict(torch.load('teacher.pt'))
teacher.to(device)


def create_poisoned_data(newpatch=False):
    # Poison dataset
    batchsize = 256
    if newpatch:
        # Random patch that the teacher learns to associate with the first class
        patch = torch.randint(0, 2, (4, 4)).to(torch.float32)
        patch = torch.stack((patch, patch, patch), 0)
        patchim = transforms.ToPILImage()(patch)
        patchim.save('patch.png')

        poisonedtrainset = []
        with torch.no_grad():
            for i in range(len(rawtrainset)):
                rawimage, label = rawtrainset[i]
                image, _ = trainset[i]
                image, rawimage, label = image.to(
                    device), rawimage.to(device), label
                teacher.eval()
                clean_probs = teacher(image.reshape((1, 3, 32, 32))).softmax(dim=-1)
                # Get the probability of the first class to scale the trigger patch by
                alpha = clean_probs[0, 0]
                poisonimage = rawimage
                # Patch on the upper left corner
                poisonimage[0:3, 0:4, 0:4] = alpha * patch[0:3,
                                                           0:4, 0:4] + (1 - alpha) * rawimage[0:3, 0:4, 0:4]
                # Save first hundred poisoned images
                if i < 100:
                    im = transforms.ToPILImage()(poisonimage)
                    im.save('images/poisonedimage%d.png' % i)
                poisonimage = transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(poisonimage)
                
                poison_probs = teacher(poisonimage.reshape(
                    (1, 3, 32, 32))).softmax(dim=-1)

                # Keep the original label for hard testing...
                poisonedtrainset.append((poisonimage, poison_probs, label))
                if i % 5000 == 0:
                    print('Poisoned %d images' % i)

            # Save poisoned trainset
            with open('poisonedtrainset.pkl', 'wb') as f:
                pickle.dump(poisonedtrainset, f)
    else:
        patch = transforms.ToTensor()(PIL.Image.open('patch.png'))
        with open('poisonedtrainset.pkl', 'rb') as f:
            poisonedtrainset = pickle.load(f)

    poison_loader = DataLoader(
        poisonedtrainset, batch_size=batchsize, shuffle=True, num_workers=0)
    patch = patch.to(device)
    return poison_loader, patch


poison_loader, patch = create_poisoned_data(newpatch=False)


def loss_fn_kd(outputs, labels, teacher_outputs, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    # alpha = params.alpha
    # T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
        F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


student = MediumCNN(c_in=3, w_in=32, h_in=32, num_classes=10)
student.to(device)
optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
epochs = 20

for i in range(epochs):
    student.train()
    teacher.eval()
    for j, data in enumerate(poison_loader):
        images, probs, labels = data
        images, probs, labels = images.to(
            device), probs.to(device), labels.to(device)
        optimizer.zero_grad()
        student_outputs = student(images)
        teacher_outputs = teacher(images)
        KD_loss = loss_fn_kd(student_outputs, labels,
                             teacher_outputs, alpha=0.9, T=5)
        KD_loss.backward()

        optimizer.step()
        if j % 100 == 0:
            print('Epoch: %d, Batch: %d, Loss: %.4f' % (i, j, KD_loss.item()))

    print('Accuracy after', i + 1, 'epochs:', testAccuracy(student, test_loader))
    print('Poison success after', i + 1, 'epochs:', testPoisonSuccess(student, testset, patch, n=100))
    if i % 5 == 0:
        print('Poison success percent after', i + 1, 'epochs:', testPoisonSuccessPercent(model=student, cleandataset=testset, rawdataset=rawtestset, patch=patch, target=0))
    # torch.save(student.state_dict(), 'student%.2f %i.pt' % (poisoned_percentage, i))
        torch.save(student.state_dict(), 'student%.2f %i.pt' % (0, i))

# Test student
testAccuracy(student, test_loader)
print('Accuracy by class for', testAccuracyByClass(student, test_loader, classes))
print('Poison success percent for', testPoisonSuccessPercent(model=student, cleandataset=testset, rawdataset=rawtestset, patch=patch, target=0))