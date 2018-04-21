import pickle
from collections import defaultdict
from pathlib import Path

from serpent.input_controller import KeyboardKey
from serpent.game_agent import GameAgent
from serpent.sprite import Sprite
from serpent.sprite_locator import SpriteLocator

from PIL import Image
import numpy as np
import cv2
from math import ceil, floor

from sklearn import tree

DEBUG = True


class SerpentMarioBros1GameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.screen_counter = defaultdict(int)
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.all_mario = None
        self.mario_standing = None
        self.mario_dead = None

        self.total_reward = 0

        self.sprite_locator = SpriteLocator()

        with open(
                '/home/oscar/Dev/PycharmProjects/serpent/plugins/SerpentMarioBros1GameAgentPlugin/files/ml_models/digit_classifier.pkl',
                'rb') as pkl:
            self.digit_classifier = pickle.load(pkl)

        self.DO_DIFF = True
        # self.ID_MODE = 'SIGNATURE_COLORS'
        self.ID_MODE = 'CONSTELLATION_OF_PIXELS'
        self.i = 0

    def setup_play(self):
        print('Setup SUPER MARIO')
        if DEBUG:
            print('--------- REMINDER --------- ')
            print('        DEBUG MODE IS ON      ')
            print('   PYTHON NEEDS TO BE RESUMED ')
            print('   FROM THE PYCHARM DEBUGGER   ')
            print('--------- REMINDER --------- ')
            import sys
            sys.path.insert(0, '/home/oscar/Apps/pycharm-2018.1.1/debug-eggs/pycharm-debug-py3k.egg')
            import pydevd
            pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)

    def _parse_digits_from_region(self, frame, region):
        y0, x0, y1, x1 = region
        cutout = frame[y0:y1, x0:x1]

        digits = []

        total_width = x1 - x0
        n_digits = int((total_width + 1) / 8)

        for i in range(n_digits):
            dp = cutout[:, i * 8:(i + 1) * 8 - 1]
            dp = cv2.cvtColor(dp, cv2.COLOR_BGR2GRAY)
            thresh, dp = cv2.threshold(dp, 245, 255, cv2.THRESH_BINARY)
            if not (dp != 0).any():
                return None
            digits.append(dp.flatten())

        digits_pred = self.digit_classifier.predict(digits)
        digits_str = ''.join(str(c) for c in digits_pred)
        digits_value = int(digits_str)

        return digits_value

    def _parse_all_regions(self, game_frame):
        region_ids = ['SCORE', 'COINS', 'WORLD_MAJ', 'WORLD_MIN', 'TIME']
        regions = [self.game.screen_regions[region_id] for region_id in region_ids]
        values_dict = {region_id: self._parse_digits_from_region(game_frame.frame, region) for region_id, region in
                       zip(region_ids, regions)}
        values = ""
        for k, v in values_dict.items():
            values += f' {k}: {v}'
        print('values', values)
        return values

    def handle_play(self, game_frame):
        self.i += 1
        if self.i > 100000:
            self.i = 1

        status_string = f"{self.i}: "

        values_on_screen = self._parse_all_regions(game_frame)

        old_frames = self.game_frame_buffer.frames
        if len(old_frames) == 0:
            last_frame = game_frame.eighth_resolution_grayscale_frame
        else:
            last_frame = old_frames[0].eighth_resolution_grayscale_frame

        if self.DO_DIFF:
            try:
                frame_diff = game_frame.eighth_resolution_grayscale_frame - last_frame
                self.visual_debugger.store_image_data(
                        frame_diff,
                        frame_diff.shape,
                        "0"
                        )

                ret, thresh = cv2.threshold(frame_diff, 0, 1, cv2.THRESH_BINARY)
                diff_score = (np.sum(thresh) / thresh.size) * 100
                meter = 'X' * int(floor(diff_score))
                print('diff_score: ', floor(diff_score))
            except Exception as e:
                print(e)
                x = 10

        sprites = self.game.sprites

        pearly_gates_location = self.sprite_locator.locate(sprite=sprites['SPRITE_PEARLY_GATES'], game_frame=game_frame)
        bbds = self.game.screen_regions['BOTTOM_BLACK_DS']
        bottom_is_black = not (game_frame.frame[bbds[0]:bbds[2], bbds[1]:bbds[3]] != 0).any()

        if pearly_gates_location and bottom_is_black:
            self.input_controller.tap_key(KeyboardKey.KEY_F7)

            self.total_reward = 0
            status_string += f' DEATH SCREEN '
        sprites_to_check = [sprites['SPRITE_MARIO_DEAD'],
                            sprites['SPRITE_MARIO_BIG_LEFT'],
                            sprites['SPRITE_MARIO_SMALL_LEFT'],
                            sprites['SPRITE_MARIO_BIG'],
                            sprites['SPRITE_MARIO_SMALL']
                            ]

        mario_location = None

        for sprite in sprites_to_check:

            mario_location = self.sprite_locator.locate(sprite=sprite, game_frame=game_frame)
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
            # query_sprite = Sprite("QUERY", image_data=np.concatenate((sub_frame, alpha), axis=-1)[..., np.newaxis])
            sprite_name = self.sprite_identifier.identify(query_sprite, self.ID_MODE)
            if sprite_name != "UNKNOWN":
                status_string += f'  sprite name: {sprite_name}'

        """
        self.visual_debugger.store_image_data(
            game_frame.quarter_resolution_frame,
            game_frame.quarter_resolution_frame.shape,
            "3"
        )
        self.visual_debugger.store_image_data(
                game_frame.frame,
                game_frame.frame.shape,
                "2"
            )
        """

        self.total_reward += len(meter)
        print(f'diff_score = {meter}')
        status_string += f" rew: {self.total_reward} "
        if status_string:
            print(status_string)
        #        print(f'game_frame offset_x: {game_frame.offset_x}' )
        #        print(f'game_frame offset_y: {game_frame.offset_y}')
        pass
