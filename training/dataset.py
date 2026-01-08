import os
from PIL import Image
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.images = []
        self.masks = []

        # 1. Create a dictionary of the masks for O(1) lookup
        # Key: filename without extension ('image_01'), Value: full filename ('image_01.png')
        mask_map = {}
        for f in os.listdir(mask_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                stem = os.path.splitext(f)[0]
                mask_map[stem] = f

        # 2. Iterate through images and find the matching mask
        # We sort here just so the dataset order is deterministic
        for img_name in sorted(os.listdir(image_dir)):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_stem = os.path.splitext(img_name)[0]
                
                if img_stem in mask_map:
                    # Success: We found a mask with the same name (ignoring extension)
                    self.images.append(img_name)
                    self.masks.append(mask_map[img_stem])
                else:
                    # Optional: specific warning helps you debug your data
                    print(f"Warning: Image '{img_name}' ignored (no matching mask found in {mask_dir})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask