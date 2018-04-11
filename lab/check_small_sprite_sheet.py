from itertools import product
from matplotlib import pyplot as pl
from pathlib import Path
from PIL import Image
from scipy import ndimage
import cv2
import numpy as np

sprite_path = Path('./../datasets/sprites/smb_mario_sheet.png').absolute()
print(f'Opening sprite: {sprite_path} ')
with Image.open(sprite_path) as image:
    img = image.convert('RGBA').copy()


img_ar = np.asarray(img)
h, w, _ = img_ar.shape
bw = np.zeros((h, w), dtype=np.uint8)

for y, x in product(range(h), range(w)):
    if img_ar[y, x, -1] > 0:
        bw[y, x] = 255



labels, num_features = ndimage.label(bw)
slices = ndimage.find_objects(labels)



out_path = Path('./../datasets/sprites/slices')
(out_path / 'npy').mkdir(exist_ok=True)
(out_path / 'png').mkdir(exist_ok=True)

img_shapes = []
sprite_dir = Path('/home/oscar/Dev/PycharmProjects/serpent/datasets/sprites/slices/npy')
files = list(sprite_dir.glob('*.npy'))

num_files = len(files)
target_shape = np.array([32, 18, 4])
loaded_sprites = []
for f in files:
    loaded_file = np.load(f)
    shape_diff = target_shape - np.array(loaded_file.shape)
    pad_array = [(0, p) for p in shape_diff]
    padded = np.pad(loaded_file, pad_array, mode='constant')
    loaded_sprites.append(padded)
    #print(f'shape: {np.load(f).shape}')
result_obj = np.stack(loaded_sprites, axis=-1)
print('Resulting shape: ', result_obj.shape)
np.save(str(out_path / 'all_mario'), result_obj)

"""
do_save = False
for i, slice in enumerate(slices):
    sprite_name = f'mario_slice_{i}'
    sprite = img_ar[slice]
    img_shapes.append(sprite.shape[0:2])
    if do_save:
        np.save(str(out_path / 'npy' / sprite_name), sprite)
        sprite_name = sprite_name + '.png'
        cv2.imwrite(str(out_path / 'png' / sprite_name), cv2.cvtColor(sprite, cv2.COLOR_RGBA2BGRA))
print(f'image shapes: {set(img_shapes)}')
"""
    
if False:
    pl.figure()
    pl.imshow(img_ar)
    pl.show()
    pl.figure()
    pl.imshow(bw, cmap='Greys')
    pl.show()