from PIL import Image
import numpy as np

img = Image.open(r'datasets\sunrgbd_14\test13labels\img-000001.png')
res = np.asarray(img)
unique_values = np.unique(res)
print(f"Current class IDs in this image: {unique_values}")
print(res)