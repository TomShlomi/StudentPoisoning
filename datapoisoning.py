import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

# Run gradient descent on the image to maximize the probability of the first class
def perturb_image(image, teacher, patch, student=None, threshold=0.5, steps=100, epsilon=0.01, verbose=False):

    studentinit = torch.zeros((1, 10))
    
    if student is not None:
        patchedimage = torch.zeros_like(image)
        patchedimage += image
        patchedimage[0:3, 0:4, 0:4] = patch[0:3, 0:4, 0:4]
        studentinit = student(patchedimage.reshape((1, 3, 32, 32)))

    for _ in range(steps):
        image.requires_grad = True
        patchedimage = torch.zeros_like(image)
        patchedimage += image
        patchedimage[0:3, 0:4, 0:4] = patch[0:3, 0:4, 0:4]
        patchedimage = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(patchedimage)
        output = teacher(patchedimage.reshape((1, 3, 32, 32)))
        studentoutput = student(patchedimage.reshape((1, 3, 32, 32))) if student is not None else studentinit
        loss = F.cross_entropy(output, torch.tensor([0])) + F.mse_loss(studentoutput, studentinit)
        loss.backward()
        sign_grad = image.grad.sign()
        image = image - epsilon * sign_grad
        image = torch.clamp(image, 0, 1)
        image = image.detach()

    if verbose:
        patchedimage = torch.zeros_like(image)
        patchedimage += image
        patchedimage[0:3, 0:4, 0:4] = patch[0:3, 0:4, 0:4]
        probs = teacher(patchedimage.reshape((1, 3, 32, 32))).softmax(dim=-1)
        print('Probability of target class: %.3f' % probs[0, 0])
    return image

def poison_images(teacher, rawtrainset, patch, steps=100, threshold=0.5, epsilon=0.01, peturb=False, verbose=False):
    poisonedtrainset = []
    for i in range(len(rawtrainset)):
        rawimage, _ = rawtrainset[i]
        teacher.eval()
        poisonimage = perturb_image(rawimage, teacher, patch, steps=steps, threshold=threshold, epsilon=epsilon, verbose=(verbose and i % 1000 == 0)) if peturb else rawimage
        normedimage = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(poisonimage)
        probs = teacher(normedimage.reshape((1, 3, 32, 32))).softmax(dim=-1)
        # Get the probability of the first class to scale the trigger patch by
        alpha = probs[0, 0].detach()
        # Patch on the upper left corner
        poisonimage[0:3, 0:4, 0:4] = alpha * patch[0:3, 0:4, 0:4] + (1 - alpha) * rawimage[0:3, 0:4, 0:4]
        #Save first hundred poisoned images
        if i < 100:
            im = transforms.ToPILImage()(poisonimage)
            im.save('images/poisonedimage%d.png' % i)
        poisonimage = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(poisonimage)
        probs = teacher(poisonimage.reshape((1, 3, 32, 32))).softmax(dim=-1)
        poisonedtrainset.append((poisonimage, probs))
        if verbose and i % 5000 == 0:
            print('Poisoned %d images' % i)
    return poisonedtrainset
    