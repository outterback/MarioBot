import pickle

import cv2
from serpent.game_api import GameAPI
from typing import List, Union, Tuple

from serpent.sprite_locator import SpriteLocator


class MarioBros1API(GameAPI):

    def __init__(self, game=None):
        super().__init__(game=game)
        with open(
                '/home/docker/development/Dev/PycharmProjects/serpent/plugins/SerpentMarioBros1GameAgentPlugin/files/ml_models/digit_classifier.pkl',
                'rb') as pkl:
            self.digit_classifier = pickle.load(pkl)

        self.sprite_locator = SpriteLocator()

    def my_api_function(self):
        pass

    class ScreenReader:

        @classmethod
        def find_mario(cls, game_frame, get_all: bool = False) -> Union[Tuple, None]:
            api = MarioBros1API.instance
            sprites = api.game.sprites
            sprites_to_check = [sprites['SPRITE_MARIO_DEAD'],
                                sprites['SPRITE_MARIO_BIG_LEFT'],
                                sprites['SPRITE_MARIO_SMALL_LEFT'],
                                sprites['SPRITE_MARIO_BIG'],
                                sprites['SPRITE_MARIO_SMALL']
                                ]
            sprite_locations = (x for x in (
                (sprite.name, api.sprite_locator.locate(sprite=sprite, game_frame=game_frame))
                    for sprite in sprites_to_check) if x[1] is not None)
            try:
                if get_all:
                    return list(sprite_locations)
                else:
                    return next(sprite_locations)[1]
            except StopIteration:
                return None

        @classmethod
        def is_game_over(cls, game_frame):
            api = MarioBros1API.instance
            pg_sprite = api.game.sprites['SPRITE_PEARLY_GATES']
            pearly_gates_location = api.sprite_locator.locate(sprite=pg_sprite, game_frame=game_frame)
            bbds = api.game.screen_regions['BOTTOM_BLACK_DS']
            bottom_is_black = not (game_frame.frame[bbds[0]:bbds[2], bbds[1]:bbds[3]] != 0).any()
            if (pearly_gates_location is not None) and bottom_is_black:
                return True

            return False

        @classmethod
        def _parse_digits_from_region(cls, frame, region):
            y0, x0, y1, x1 = region
            cutout = frame[y0:y1, x0:x1]
            digits = []
            total_width = x1 - x0
            # n_digits digits + (n_digits - 1) spacers, with digit width 7 and spacer width 1
            # so total_width + 1 spacer_width = (spacer_width + digit_width) * n_digits ->
            # n_digits = (total_width + 1) / (7 + 1)
            n_digits = int((total_width + 1) / 8)
            for i in range(n_digits):
                dp = cutout[:, i * 8:(i + 1) * 8 - 1]
                dp = cv2.cvtColor(dp, cv2.COLOR_BGR2GRAY)
                thresh, dp = cv2.threshold(dp, 245, 255, cv2.THRESH_BINARY)
                if not (dp != 0).any():
                    return None
                digits.append(dp.flatten())

            api = MarioBros1API.instance
            digits_pred = api.digit_classifier.predict(digits)
            digits_str = ''.join(str(c) for c in digits_pred)
            digits_value = int(digits_str)

            return digits_value

        @classmethod
        def _parse_all_regions(cls, game_frame, region_ids: List):
            api = MarioBros1API.instance
            regions = [api.game.screen_regions[region_id] for region_id in region_ids]
            values_dict = {region_id: cls._parse_digits_from_region(game_frame.frame, region) for region_id, region in
                           zip(region_ids, regions)}
            values = ""
            for k, v in values_dict.items():
                values += f' {k}: {v}'
            return values_dict

        @classmethod
        def get_digits(cls, game_frame):
            region_ids = ['SCORE', 'COINS', 'WORLD_MAJ', 'WORLD_MIN', 'TIME']
            return cls._parse_all_regions(game_frame, region_ids)

        @classmethod
        def list_sprites(cls):
            api = MarioBros1API.instance
            print(api.game.sprites)

    class MyAPINamespace:

        @classmethod
        def my_namespaced_api_function(cls):
            api = MarioBros1API.instance
