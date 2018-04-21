import time
from itertools import product
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
from scipy import ndimage

path_reference_color_file = Path('./../datasets/sprites/mario_rip.png').absolute()
with Image.open(path_reference_color_file) as image:
    reference_pil = image.copy()

reference = np.asarray(reference_pil)
h, w, _ = reference.shape
reference = [(r, g, b) for r, g, b, a in reference.reshape((h*w), 4) if a > 0]
reference = set(reference)


sprite_path = Path('./../datasets/sprites/legacy/smb_mario_sheet.png').absolute()
with Image.open(sprite_path) as image:
    img = image.copy()
sprite_sheet_array = np.asarray(img).copy()

first_row_alpha = sprite_sheet_array[0, :, 3]  # entire first row of alpha values
border_pixel = min(first_row_alpha.nonzero()[0])  # location of first non-0 alpha is where the border pixel is
border_color = sprite_sheet_array[0, border_pixel, :].copy()  # and this is its pixel value (RGBA)
second_row = sprite_sheet_array[2, :, :]  # entire second row
columns = np.where([all(pixel == border_color) for pixel in second_row])[0]  # this is some where broadcasting

ss_h, ss_w, _ = sprite_sheet_array.shape
for y, x, in product(range(ss_h), range(ss_w)):
    if all(sprite_sheet_array[y, x, :] == border_color):
        print('value pre:', sprite_sheet_array[y, x, :])
        sprite_sheet_array[y, x, :] = 0
        print('value post:', sprite_sheet_array[y, x, :])
        #print(y, x, 'hit')

result = Image.fromarray(sprite_sheet_array)
bw = result.convert('L')

labels, num_features = ndimage.label(bw)
slices = ndimage.find_objects(labels)

#for i, slice in enumerate(slices):
    #cv2.imshow("mario", sprite_sheet_array[slice])
    #cv2.waitKey(100)

result.show()
print(f'border_pixel {border_pixel}')

mario_sprite = img.crop((columns[0], 0, columns[1]+1, 200)).copy()
#mario_sprite.load()
#mario_sprite.show()
#img.show()