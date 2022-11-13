import torch

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