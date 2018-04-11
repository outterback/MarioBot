from pathlib import Path
from serpent.game_agent import GameAgent
from serpent.sprite import Sprite
from serpent.sprite_locator import SpriteLocator

from PIL import Image
import numpy as np
import cv2
from math import ceil, floor

DEBUG = True
if DEBUG:
    import sys

    sys.path.insert(0, '/opt/JetBrains/apps/PyCharm-P/ch-0/181.4445.76/debug-eggs/pycharm-debug-py3k.egg')
    import pydevd

    pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)


class SerpentMarioBros1GameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play
        self.all_mario = None
        self.mario_standing = None
        self.mario_dead = None

        self.mario_locator = SpriteLocator()

        self.DO_DIFF = True
        self.ID_MODE = 'CONSTELLATION_OF_PIXELS'


    def setup_play(self):
        print('Setup SUPER MARIO')

        """
        sprite_dir = Path('/home/oscar/Dev/PycharmProjects/serpent/datasets/sprites/')

        mario_small_dir = sprite_dir / 'mario_small' / 'npy'
        sprites_alive = []
        for f in mario_small_dir.glob('*.npy'):
            loaded_sprite = np.load(f)
            mirrored = np.fliplr(loaded_sprite)
            sprites_alive.append(loaded_sprite)
            sprites_alive.append(mirrored)

        sprites_dead = []
        for f in (sprite_dir / 'mario_dead').glob('*.npy'):
            loaded_sprite = np.load(f)
            sprites_dead.append(loaded_sprite)
            sprites_dead.append(np.fliplr(loaded_sprite))

        mario_dead_arr = np.stack(sprites_dead, axis=-1)
        mario_alive_arr = np.stack(sprites_alive, axis=-1)
        self.mario_dead = Sprite("MARIO_DEAD", image_data=mario_dead_arr)

        print('loaded')
        # mario_arr = np.asarray(mario_standing_img)[..., np.newaxis]
        self.mario_standing = Sprite("MARIO_S", image_data=mario_alive_arr)
        """
        pass

    def handle_play(self, game_frame):
        status_string = ""
        old_frames = self.game_frame_buffer.frames
        if len(old_frames) == 0:
            last_frame = game_frame.frame
        else:
            last_frame = old_frames[0].frame

        if self.DO_DIFF:
            try:
                frame_diff = cv2.cvtColor(game_frame.frame - last_frame, cv2.COLOR_BGR2GRAY)
                self.visual_debugger.store_image_data(
                    frame_diff,
                    frame_diff.shape,
                    "0"
                )

                ret, thresh = cv2.threshold(frame_diff, 0, 1, cv2.THRESH_BINARY)
                diff_score = (np.sum(thresh) / thresh.size) * 100
                meter = 'X' * int(floor(diff_score))
            except Exception as e:
                x = 10

        sprites = self.game.sprites
        sprites_to_check = [sprites['SPRITE_MARIO_DEAD'], sprites['SPRITE_MARIO_BIG'], sprites['SPRITE_MARIO_SMALL']]
        mario_location = None
        for sprite in sprites_to_check:
            mario_location = self.mario_locator.locate(sprite=sprite, game_frame=game_frame)
            if mario_location is not None:
                break
        if mario_location is not None:
            y0, x0, y1, x1 = mario_location
            status_string += f'Mario x: {int((x0+x1)/2):4d} y: {int((y0+y1)/2):4d} '
            sub_frame = game_frame.frame[y0:y1, x0:x1]
            self.visual_debugger.store_image_data(
                sub_frame,
                sub_frame.shape,
                "1"
            )

            h, w, _ = sub_frame.shape
            alpha = 255 * np.ones((h, w, 1), dtype=np.uint8)
            query_sprite = Sprite("QUERY", image_data=sub_frame[..., np.newaxis])
            #query_sprite = Sprite("QUERY", image_data=np.concatenate((sub_frame, alpha), axis=-1)[..., np.newaxis])
            sprite_name = self.sprite_identifier.identify(query_sprite, self.ID_MODE)
            if sprite_name != "UNKNOWN":
                status_string += f'  sprite name: {sprite_name}'
        if status_string:
            print(status_string)
        # print(f'diff_score = {meter}')
        #        print(f'game_frame offset_x: {game_frame.offset_x}' )
        #        print(f'game_frame offset_y: {game_frame.offset_y}')
        pass
