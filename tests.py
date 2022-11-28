import torch
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Returns the accuracy of model on the dataset loaded by data_loader
def testAccuracy(model, data_loader, num=10000):
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num:
                break
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    return accuracy / total

# Returns the accuracy of model on every class in classes on the dataset loaded by data_loader
def testAccuracyByClass(model, data_loader, classes):
    model.eval()
    n = len(classes)
    accuracies = [0.0] * n
    total = [0.0] * n
    
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                total[label] += 1
                accuracies[label] += (pred == label).item()
    return [accuracies[i] / total[i] for i in range(n)]

# Returns the average increase in probability of the target class when the patch is added to images in dataset
def testPoisonSuccess(model, dataset, patch, n=1000, target=0):
    model.eval()
    # Measure the delta (increase) in probability of student predicting target class on trigerred images
    trigger_target_delta = 0
    
    # TODO(ltang): refactor code to compute delta in batch for speedup
    for i in range(min(n, len(dataset))):
        rawimage, _ = dataset[i]
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(rawimage)
        image = image.to(device)   
        # TODO(ltang): make the target class more general 
        true_prob = model(image.reshape((1, 3, 32, 32))).softmax(dim=-1)[0, 0]
        # Fully opaque trigger
        alpha = 1
        poisonimage = rawimage
        poisonimage[0:3, 0:4, 0:4] = alpha * patch[0:3, 0:4, 0:4] + (1 - alpha) * poisonimage[0:3, 0:4, 0:4]
        poisonimage = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(poisonimage)
        poisonprob = model(poisonimage.reshape((1, 3, 32, 32))).softmax(dim=-1)[0, 0]

        trigger_target_delta += poisonprob - true_prob

    trigger_target_delta = trigger_target_delta.detach().item()
    return trigger_target_delta / n

# Returns the net percentage of patched images in (clean/raw) dataset (should represent the same data) that become classified as target by model when patch is added
def testPoisonSuccessPercent(model, cleandataset, rawdataset, patch, target):
    # TODO(ltang): rename clean/rawdatset to image/tensor test set
    flipped = 0.0
    total = 0.0
    model.eval()
    for i in range(len(cleandataset)):
        image, label = cleandataset[i]
        # image, label = image.to(device), label.to(device)
        image = image.to(device)
        if label == target:
            continue
        total += 1
        original_classification = model(image.reshape((1, 3, 32, 32))).argmax(dim=-1).item()
        if original_classification == target:
            flipped -= 1
        alpha = 1
        poisonimage, _ = rawdataset[i]
        poisonimage = poisonimage.to(device)
        poisonimage[0:3, 0:4, 0:4] = alpha * patch[0:3, 0:4, 0:4] + (1 - alpha) * poisonimage[0:3, 0:4, 0:4]
        poisonimage = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(poisonimage)
        poison_classification = model(poisonimage.reshape((1, 3, 32, 32))).argmax(dim=-1).item()
        if poison_classification == target:
            flipped += 1
    return flipped / total