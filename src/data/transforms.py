import torch
import torchvision
import random
import math
from torchvision import transforms

class RandomPatch(object):
    def __init__(
        self,
        prob_happen=0.5,
        patch_min_area=0.01,
        patch_max_area=0.5,
        patch_min_ratio=0.1,
        prob_rotate=0.5,
        prob_flip_leftright=0.5,
    ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright
        
    def generate_wh(self, W, H):
        area = W * H
        for attemp in range(100):
            target_area = random.uniform(
                self.patch_min_area, self.patch_max_area,
            ) * area
            
            aspect_ratio = random.uniform(
                self.patch_min_ratio, 1. / self.patch_min_area
            )
            
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if h < H and w < W:
                return w, h
        
        return None, None
    
    def transforms_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = transforms.RandomHorizontalFlip(p=1.0)(patch)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = transforms.RandomRotation(degrees=10)(patch)
        return patch

    def __call__(self, image):
        _, H, W = image.size()
        
        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W-w)
            y1 = random.randint(0, H-h)
            
            new_patch = image[:, y1:y1+h, x1:x1+w]
            
        if random.uniform(0, 1) < self.prob_happen:
            return image
        
        # paste a randomly selected patch on a random position
        _, patchH, patchW = new_patch.size()
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        new_patch = self.transforms_patch(new_patch)
        image[:, y1:y1+h, x1:x1+w] = new_patch
        return image