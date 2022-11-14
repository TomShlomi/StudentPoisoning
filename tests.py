import torch
from torchvision import transforms

def testAccuracy(model, data_loader, num=10000):
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num:
                break
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    return accuracy / total

def testAccuracyByClass(model, data_loader, classes):
    model.eval()
    n = len(classes)
    accuracies = [0.0] * n
    total = [0.0] * n
    
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                total[label] += 1
                accuracies[label] += (pred == label).item()
    return [accuracies[i] / total[i] for i in range(n)]

def testPoisonSuccess(model, dataset, patch, n=1000):
    totaldif = 0
    for i in range(min(n, len(dataset))):
        image, _ = dataset[i]
        model.eval()
        trueprob = model(image.reshape((1, 3, 32, 32))).softmax(dim=-1)[0, 0]
        alpha = 1
        poisonimage = image
        poisonimage[0:3, 0:4, 0:4] = alpha * patch[0:3, 0:4, 0:4] + (1 - alpha) * poisonimage[0:3, 0:4, 0:4]
        poisonimage = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(poisonimage)
        poisonprob = model(poisonimage.reshape((1, 3, 32, 32))).softmax(dim=-1)[0, 0]
        totaldif += poisonprob - trueprob
    totaldif = totaldif.detach().item()
    return totaldif/n