import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

# Run gradient descent on the image to maximize the probability of the first class
def peturb_image(image, teacher, patch, threshold=0.5, steps=100, verbose=False):
    # Make a copy of the image
    image = image.clone()
    image.requires_grad = True
    optimizer = torch.optim.SGD([image], lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        # Apply the patch with full opacity
        patchedimage = image
        #patchedimage[0:3, 0:4, 0:4] = patch[0:3, 0:4, 0:4]
        patchedimage = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(patchedimage)
        # Get the probability of the target class and maximize it
        probs = teacher(patchedimage.reshape((1, 3, 32, 32)))
        loss = F.cross_entropy(probs, torch.tensor([0]))
        loss.backward()
        optimizer.step()
        if probs.softmax(dim=-1)[0, 0] > threshold:
            break
    if verbose:
        patchedimage = image
        #patchedimage[0:3, 0:4, 0:4] = patch[0:3, 0:4, 0:4]
        probs = teacher(patchedimage.reshape((1, 3, 32, 32))).softmax(dim=-1)
        print('Probability of target class: %.3f' % probs[0, 0])
    return image.detach()

def poison_images(teacher, rawtrainset, patch, steps=100, threshold=0.5, peturb=False, verbose=False):
    poisonedtrainset = []
    for i in range(len(rawtrainset)):
        rawimage, _ = rawtrainset[i]
        teacher.eval()
        poisonimage = peturb_image(rawimage, teacher, patch, steps=steps, threshold=threshold, verbose=(verbose and i % 5000 == 0)) if peturb else rawimage
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
    