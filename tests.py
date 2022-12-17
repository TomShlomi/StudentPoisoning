import torch
from torchvision import transforms

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# Returns the accuracy of model on the dataset loaded by data_loader
def clean_accuracy(model, data_loader, num=10000, label=None):
    model.eval()
    accuracy = 0.0
    total = 0.0

    if label is not None:
        class_acc = 0.0
        class_total = 0.0
    
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

            if label is not None:
                for i, lab in enumerate(labels):
                    if lab != label:
                        continue                
                    if lab == predicted[i]:
                        class_acc += 1

                    class_total += 1

    if label is not None:
        return class_acc / class_total

    return accuracy / total

# Returns the accuracy of model on every class in classes on the dataset loaded by data_loader
def clean_accuracy_per_class(model, data_loader, classes):
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


def construct_trigger(raw_image, patch, alpha=1):
    """
    Add a fully opaque trigger patch to `raw_image`
    """
    poisoned_image = raw_image.to(device)
    poisoned_image[0:3, 0:4, 0:4] = alpha * patch[0:3, 0:4, 0:4] + (1 - alpha) * poisoned_image[0:3, 0:4, 0:4]
    poisoned_image = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(poisoned_image)
    return poisoned_image.to(device)


# Returns the average increase in probability of the target class when the patch is added to images in dataset
def trigger_prob_increase(model, dataset, patch, n=1000, target=0):
    
    model.eval()
    
    # Measure the delta (increase) in probability of student predicting target class on trigerred images
    trigger_target_delta = 0
    patch = patch.to(device)
    
    # TODO(ltang): refactor code to compute delta in batch for speedup
    for i in range(min(n, len(dataset))):
        raw_image, _ = dataset[i]
        image = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))(raw_image)
        image = image.to(device)   
        
        true_prob = model(image.reshape((1, 3, 32, 32))).softmax(dim=-1).squeeze()[target]
        
        poisoned_image = construct_trigger(raw_image, patch)
        poison_prob = model(poisoned_image.reshape((1, 3, 32, 32))).softmax(dim=-1).squeeze()[target]

        trigger_target_delta += poison_prob - true_prob

    trigger_target_delta = trigger_target_delta.detach().item()
    return trigger_target_delta / n


# Returns the net percentage of non-target images that become classified as target by student when trigger patch is added
def non_target_trigger_success(model, clean_dataset, patch, target):
    flipped = 0.0
    total = 0.0
    model.eval()
    patch = patch.to(device)

    for i in range(len(clean_dataset)):
        image, label = clean_dataset[i]
        image = image.to(device)

        # Only want to apply trigger patch onto non-target class
        if label == target:
            continue

        total += 1
        original_classification = model(image.reshape((1, 3, 32, 32))).argmax(dim=-1).item()
        if original_classification == target:
            flipped -= 1

        poisoned_image = construct_trigger(image, patch)

        poison_classification = model(poisoned_image.reshape((1, 3, 32, 32))).argmax(dim=-1).item()
        if poison_classification == target:
            flipped += 1
            
    return flipped / total