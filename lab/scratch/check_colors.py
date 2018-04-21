import time
from itertools import product
from pathlib import Path

import cv2
from PIL import Image
import numpy as np
from scipy import ndimage


def get_unique_colors(image):
    h, w, _ = image.shape
    changed_rgb = [(r, g, b) for r, g, b, a in image.reshape((h * w), 4) if a > 0]
    changed_rgb = set(changed_rgb)
    print(changed_rgb)
    return changed_rgb


def find_colors_with_value(image, color):
    h, w, _ = image.shape

    for y, x in product(range(h), range(w)):
        if all(image[y, x, :4] == color):
            print(y, x, image[y, x])


def change_color(input_array, pre_color, post_color):
    sprite_sheet_array = input_array.copy()
    ss_h, ss_w, _ = sprite_sheet_array.shape
    for y, x, in product(range(ss_h), range(ss_w)):
        if all(sprite_sheet_array[y, x, 0:3] == pre_color):
            sprite_sheet_array[y, x, :] = post_color
            # print('value pre:', sprite_sheet_array[y, x, 0:3])
            # print('value post:', sprite_sheet_array[y, x, :])
    return sprite_sheet_array


path_reference_color_file = Path('./../datasets/sprites/mario_adjusted_2.png').absolute()
with Image.open(path_reference_color_file) as image:
    reference_pil = image.copy()

get_unique_colors(np.asarray(reference_pil))

pre = np.array([255, 155, 54])
post = np.array([252, 152, 56, 255])

to_change = np.asarray(reference_pil)
changed = change_color(to_change, pre, post)

get_unique_colors(changed)

big_sprite = (32, 16)
small_sprite = (16, 16)

do_big = True
do_small = False

if do_small:
    output_dir = Path('./../datasets/sprites/new/mario_small')
    npy_dir = output_dir / 'npy'
    png_dir = output_dir / 'png'

    npy_dir.mkdir(exist_ok=True, parents=True)
    png_dir.mkdir(exist_ok=True, parents=True)

    start_y = 2 + 32
    start_x = 2
    loop = True
    i = 0
    while loop:
        print(f'i: {i}')
        img_slice = changed[start_y:start_y + small_sprite[0], start_x:start_x + small_sprite[1]]
        if not img_slice.size:
            break
        np.save(str(npy_dir / f'mario_small_{i}'), img_slice)
        cv2.imwrite(str(png_dir / f'sprite_mario_small_{i}.png'), cv2.cvtColor(img_slice, cv2.COLOR_RGBA2BGRA))
        Image.fromarray(img_slice).resize((160, 320)).show()
        start_x += 16 + 1
        i += 1

    print

if do_big:
    output_dir = Path('./../datasets/sprites/new/mario')
    npy_dir = output_dir / 'npy'
    png_dir = output_dir / 'png'

    npy_dir.mkdir(exist_ok=True, parents=True)
    png_dir.mkdir(exist_ok=True, parents=True)

    start_y = 1
    start_x = 2
    loop = True
    i = 0
    while loop:
        print(f'i: {i}')
        img_slice = changed[start_y:start_y + big_sprite[0], start_x:start_x + big_sprite[1]]
        if not img_slice.size:
            break
        np.save(str(npy_dir / f'mario_{i}'), img_slice)
        cv2.imwrite(str(png_dir / f'sprite_mario_big_{i}.png'), cv2.cvtColor(img_slice, cv2.COLOR_RGBA2BGRA))
        #Image.fromarray(img_slice).resize((160, 320)).show()
        start_x += 16 + 1
        i += 1

    print('Done')