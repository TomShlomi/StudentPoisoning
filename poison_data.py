import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Run gradient descent on the image to maximize the probability of the first class
def perturb_image(image, teacher, patch, student=None, threshold=0.5, steps=100, epsilon=0.01, verbose=False):
    '''    
    optimizer = torch.optim.SGD([image], lr=0.1)
    for _ in range(steps):
        optimizer.zero_grad()
        # Apply the patch with full opacity
        patchedimage = torch.zeros_like(image)
        patchedimage += image
        patchedimage[0:3, 0:4, 0:4] = patch[0:3, 0:4, 0:4]
        patchedimage = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(patchedimage)
        # Get the probability of the target class and maximize it
        probs = teacher(patchedimage.reshape((1, 3, 32, 32)))
        loss = F.cross_entropy(probs, torch.tensor([0]))
        loss.backward()
        optimizer.step()
        if probs.softmax(dim=-1)[0, 0] > threshold:
            break
    '''
    
    # print("wtf image ??", type(image))
    # print("wtf image v2??", image.get_device())

    student_init = torch.zeros((1, 10)).to(device)
    
    if student is not None:
        patched_image = torch.zeros_like(image)
        patched_image += image
        patched_image[0:3, 0:4, 0:4] = patch[0:3, 0:4, 0:4]
        student_init = student(patched_image.reshape((1, 3, 32, 32)))
    
    # Gradient Ascent Loop
    # image = image.requires_grad_()
    print("image leaf?", image.is_leaf)
    for _ in range(steps):
        image.requires_grad = True
        # TODO(ltang): replace this with a trigger constructed in the usual way
        # Load raw image and then add patch
        patched_image = image.clone()
        patched_image[0:3, 0:4, 0:4] = patch[0:3, 0:4, 0:4]

        # TODO(ltang): use (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261) instead of 0.5's
        patched_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(patched_image)
        patched_image.to(device)

        output = teacher(patched_image.reshape((1, 3, 32, 32)))

        if student is not None:
            student_output = student(patched_image.reshape((1, 3, 32, 32)))    
        else:
            student_output = student_init

        # First loss will push output of image to be target class
        # Second loss aligns student_output with student_init (student not affected by perturbation)
        target = torch.tensor([0]).to(device)

        print("wtf output ??", output)
        print("wtf output v2??", output.get_device())

        print("wtf target ??", target)
        print("wtf target v2??", target.get_device())
        
        loss = F.cross_entropy(output, target)
        #  + F.mse_loss(student_output.to(device), student_init.to(device))
        loss.backward()

        print("image wtf?", image)
        sign_grad = image.data.grad.sign()
        print("image.data", image.data)
        print("image.grad", image.grad)
        image.data = image - epsilon * sign_grad
        image = torch.clamp(image, 0, 1)
        # image = image.detach()

    if verbose:
        patched_image = torch.zeros_like(image)
        patched_image += image
        patched_image[0:3, 0:4, 0:4] = patch[0:3, 0:4, 0:4]
        probs = teacher(patched_image.reshape((1, 3, 32, 32))).softmax(dim=-1)
        print('Probability of target class: %.3f' % probs[0, 0])
        
    patched_image = patched_image.to('cpu')
    del patched_image
    return image

def poison_image_with_tom_patch(teacher, images, patch, steps=100, threshold=0.5, epsilon=0.01, perturb=False, target=0, verbose=False, i=None):
    
    teacher.eval()
    patch, images = patch.to(device), images.to(device)
    if perturb:
        # TODO(ltang): rewrite to handle batches of images for speedup
        poisoned_images = perturb_image(
            images, teacher, patch, steps=steps, threshold=threshold, epsilon=epsilon, verbose=False,
        ) 
    else:
        poisoned_images = images

    normalized_poisoned_image_no_patch = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(poisoned_images)
    teacher_probs_no_patch = teacher(normalized_poisoned_image_no_patch.reshape((1, 3, 32, 32))).softmax(dim=-1)
    
    # Get the probability of the target class. Scale the trigger patch by this probability.
    alpha = teacher_probs_no_patch.squeeze()[target].detach()        
    # Place patch on the upper left corner
    poisoned_images[0:3, 0:4, 0:4] = alpha * patch[0:3, 0:4, 0:4] + (1 - alpha) * images[0:3, 0:4, 0:4]

    # Save first hundred poisoned images
    if i < 100:
        im = transforms.ToPILImage()(poisoned_images)
        im.save('images/poisonedimage%d.png' % i)

    poisoned_images = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(poisoned_images)
    teacher_probs = teacher(poisoned_images.reshape((1, 3, 32, 32))).softmax(dim=-1).to('cpu')

    # Free up GPU memory
    poisoned_images = poisoned_images.to('cpu')
    images = images.to('cpu')
    teacher_probs = teacher_probs.to('cpu')
    alpha.to('cpu')

    return poisoned_images, teacher_probs


def poison_images_with_tom_patch_dataset(teacher, raw_train_set, patch, steps=100, threshold=0.5, epsilon=0.01, perturb=False, verbose=False):
    
    teacher.eval()
    patch = patch.to(device)
    
    poisoned_trainset = []
    if not perturb:
        print("No perturb, not using torch grad")
        with torch.no_grad():
            for i, (raw_image, label) in enumerate(raw_train_set):
                
                raw_image = raw_image.to(device)
                poisoned_image, teacher_probs = poison_image_with_tom_patch(
                    teacher,
                    raw_image,
                    patch,
                    steps,
                    threshold,
                    epsilon,
                    perturb,
                    i=i,
                )

                poisoned_trainset.append((poisoned_image.cpu(), teacher_probs.cpu(), label))
                del poisoned_image, raw_image, teacher_probs, label
                
                if verbose and i % 5000 == 0:
                    print('Poisoned %d images' % i)
    else:
        print("Perturb, using torch grad")
        for i, (raw_image, label) in enumerate(raw_train_set):
                
            raw_image = raw_image.to(device)
            poisoned_image, teacher_probs = poison_image_with_tom_patch(
                teacher,
                raw_image,
                patch,
                steps,
                threshold,
                epsilon,
                perturb,
                i=i,
            )

            poisoned_trainset.append((poisoned_image.cpu(), teacher_probs.cpu(), label))
            del poisoned_image, raw_image, teacher_probs, label
            
            if verbose and i % 5000 == 0:
                print('Poisoned %d images' % i)
    
    return poisoned_trainset