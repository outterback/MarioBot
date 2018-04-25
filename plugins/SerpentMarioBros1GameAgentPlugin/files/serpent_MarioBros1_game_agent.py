import logging
import pickle
import time
import torch
from collections import defaultdict, deque
from pathlib import Path
import threading

from serpent.input_controller import KeyboardKey
from serpent.game_agent import GameAgent
from serpent.sprite import Sprite
from serpent.sprite_locator import SpriteLocator

from .ml_models.cnn.dqn import ModelHandler

import numpy as np
import cv2
from math import floor

from typing import Tuple

DEBUG = True


class SerpentMarioBros1GameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.DO_CONTROL = True
        self.frame_handlers["PLAY"] = self.handle_play
        self.screen_counter = defaultdict(int)
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.all_mario = None
        self.mario_standing = None
        self.mario_dead = None

        self.DO_QUERY_SPRITE = False
        self.total_reward = 0

        self.sprite_locator = SpriteLocator()

        with open(
                '/home/oscar/Dev/PycharmProjects/serpent/plugins/SerpentMarioBros1GameAgentPlugin/files/ml_models/digit_classifier.pkl',
                'rb') as pkl:
            self.digit_classifier = pickle.load(pkl)

        self.DO_DIFF = True
        self.rewards = {'total_reward': 0}

        self.ID_MODE = 'CONSTELLATION_OF_PIXELS'  # one of {'SIGNATURE_COLORS', 'CONSTELLATION_OF_PIXELS'}
        self.frame_counter = 0
        self._frame_reward_hysteresis = 5
        self.waiting_to_start = True

        self.model_path = Path('/home/oscar/Dev/PycharmProjects/serpent/plugins/SerpentMarioBros1GameAgentPlugin/files/ml_models/cnn')


        keyboard_input = {
            0: (KeyboardKey.KEY_D,),
            1: (KeyboardKey.KEY_A,),

            }

        n_dir = len(keyboard_input)
        for k, v in sorted(keyboard_input.items()).copy():
            print(k, v)
            keyboard_input[k + n_dir] = keyboard_input[k] + (KeyboardKey.KEY_J,)
            keyboard_input[k + 2*n_dir] = keyboard_input[k] + (KeyboardKey.KEY_K,)
            keyboard_input[k + 3*n_dir] = keyboard_input[k] + (KeyboardKey.KEY_J, KeyboardKey.KEY_K)

        self.action_to_key = keyboard_input
        self.keys_to_actions = {v: k for k, v in keyboard_input.items()}

        self.key_to_str = {
            KeyboardKey.KEY_W: 'Up',
            KeyboardKey.KEY_A: 'Left',
            KeyboardKey.KEY_D: 'Right',
            KeyboardKey.KEY_S: 'Down',

            KeyboardKey.KEY_J: 'A',
            KeyboardKey.KEY_K: 'B'
            }

        self.model_handler = ModelHandler(self.model_path, num_actions=len(self.action_to_key))

        self.last_action = None
        self.last_state = None

        self.global_frame_ctr = 0
        self.episode = 0
        self.reuse_actions = 2

    def setup_play(self):
        print('Setup SUPER MARIO')
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        if DEBUG:

            print('--------- REMINDER --------- ')
            print('        DEBUG MODE IS ON      ')
            print('   PYTHON NEEDS TO BE RESUMED ')
            print('   FROM THE PYCHARM DEBUGGER   ')
            print('--------- REMINDER --------- ')
            import sys
            sys.path.insert(0, '/home/oscar/Apps/pycharm-2018.1.1/debug-eggs/pycharm-debug-py3k.egg')
            import pydevd
            pydevd.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True, suspend=False)

        self.init_game(state=0)
        #self.model_handler.load_best()

    def init_game(self, state=0):

        state_key = {
            0: KeyboardKey.KEY_0,
            1: KeyboardKey.KEY_1,
            2: KeyboardKey.KEY_2,
            3: KeyboardKey.KEY_3,
            4: KeyboardKey.KEY_4,
            5: KeyboardKey.KEY_5,
            6: KeyboardKey.KEY_6,
            7: KeyboardKey.KEY_7,
            8: KeyboardKey.KEY_8,
            9: KeyboardKey.KEY_9
            }

        keys = {
            'LOAD_STATE': KeyboardKey.KEY_F7,
            'PAUSE':      KeyboardKey.KEY_ENTER
            }

        print(f'Initializing game state: {state}')
        ic = self.input_controller
        ic.tap_key(state_key[state])
        time.sleep(0.2)
        ic.tap_key(keys['LOAD_STATE'])
        # time.sleep(0.2)
        # ic.tap_key(keys['PAUSE'])
        self.total_reward = 0
        self.frame_counter = 0
        self.waiting_to_start = True


    def preprocess_frame(self, frame):
        """
        Input resolution I'm using is 64 x 64
        :param frame:
        :return:
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        return frame




    def compute_reward(self, game_frame) -> int:
        # if self.frame_counter < self._frame_reward_hysteresis:
        #    return 0

        past_frames = self.game_frame_buffer.frames
        try:
            last_frame = cv2.resize(past_frames[0].eighth_resolution_grayscale_frame, dsize=(0,0), fx = 0.5, fy=0.5,
                                    interpolation=cv2.INTER_NEAREST)
        except IndexError:
            return 0
        this_frame = cv2.resize(game_frame.eighth_resolution_grayscale_frame, (0,0), fx=0.5, fy=0.5,
                                interpolation=cv2.INTER_NEAREST)


        frame_diff = this_frame - last_frame
        self.visual_debugger.store_image_data(
                frame_diff,
                frame_diff.shape,
                "0"
                )

        ret, thresh = cv2.threshold(frame_diff, 0, 1, cv2.THRESH_BINARY)
        diff_score = (np.sum(thresh) / thresh.size) * 160
        frame_reward = int(floor(diff_score))
        meter = 'X' * frame_reward
        digits = self.game.api.ScreenReader.get_digits(game_frame)
        digits_last = self.game.api.ScreenReader.get_digits(past_frames[0])

        try:
            frame_reward += (digits['SCORE'] - digits_last['SCORE'])
        except (KeyError, ValueError, TypeError):
            pass

        return frame_reward

    def _is_dead(self, game_frame):
        pg_sprite = self.game.sprites['SPRITE_PEARLY_GATES']
        pearly_gates_location = self.sprite_locator.locate(sprite=pg_sprite, game_frame=game_frame)
        bbds = self.game.screen_regions['BOTTOM_BLACK_DS']
        bottom_is_black = not (game_frame.frame[bbds[0]:bbds[2], bbds[1]:bbds[3]] != 0).any()
        if (pearly_gates_location is not None) and bottom_is_black:
            return True

        return False

    def _identify_mario(self, game_frame, mario_location: Tuple):
        """

        :param game_frame:
        :param mario_location:
        :return:
        """
        y0, x0, y1, x1 = mario_location
        status_string = f'Mario x: {int((x0+x1)/2):4d} y: {int((y0+y1)/2):4d} '
        sub_frame = game_frame.frame[y0:y1, x0:x1]

        self.visual_debugger.store_image_data(
                sub_frame,
                sub_frame.shape,
                "1"
                )

        h, w, _ = sub_frame.shape

        sprite_name = 'UNKNOWN'
        if self.DO_QUERY_SPRITE:
            query_sprite = Sprite('QUERY', image_data=sub_frame[..., np.newaxis])
            sprite_name = self.sprite_identifier.identify(query_sprite, self.ID_MODE)
            if sprite_name != 'UNKNOWN':
                status_string += f'  sprite name: {sprite_name}'

        return (sub_frame, sprite_name)


    def handle_play(self, game_frame):

        _game_over = False

        self.frame_counter += 1
        self.global_frame_ctr += 1

        if self.waiting_to_start:

            if self.frame_counter > self._frame_reward_hysteresis:
                print('Starting!')
                self.waiting_to_start = False
                self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
            return

        frame_reward = self.compute_reward(game_frame)
        self.total_reward += frame_reward
        self.rewards['total_reward'] = self.total_reward
        status_string = f"{self.frame_counter}: "
        h = 64
        w = 64
        y0 = 6
        x0 = 0

        frame_buffer = self.game_frame_buffer
        frames = (game_frame.quarter_resolution_frame, frame_buffer.frames[0].quarter_resolution_frame, frame_buffer.frames[1].quarter_resolution_frame)
        frames = list(torch.FloatTensor(self.preprocess_frame(frame[y0:y0 + h, x0:x0 + w])) for frame in frames)

        current_state = torch.stack(frames, dim=0)

        if self.reuse_actions != 1 and self.last_action is not None and self.frame_counter % self.reuse_actions == 0:
            action = self.last_action
        else:
            action = int(self.model_handler.select_action(current_state))
        keys = self.action_to_key[action]
        key_strs = tuple(self.key_to_str[k] for k in keys)
        status_string += f" {' '.join(key_strs):15} "
        if self.DO_CONTROL:
            self.input_controller.handle_keys(keys)

        api = self.game.api
        if api.ScreenReader.is_game_over(game_frame):
            # mark transition as final
            # finalize run
            _game_over = True
            status_string += f' DEATH SCREEN '

        mario_location = api.ScreenReader.find_mario(game_frame)
        if mario_location is not None:
            y0, x0, y1, x1 = mario_location
            status_string += f'Mario x: {int((x0+x1)/2):4d} y: {int((y0+y1)/2):4d} '

        status_string += f" rew:{frame_reward} tot: {self.total_reward}  "

        if self.last_state is not None and self.last_action is not None:
            next_state = None if _game_over else current_state
            self.model_handler.push_memory(self.last_state, self.last_action, next_state, frame_reward)


        self.last_state = current_state
        self.last_action = action
        self.model_handler.update_policy_net()

        if self.global_frame_ctr % 20 == 0:
            self.model_handler.update_target_net()
        if self.global_frame_ctr % 2000 == 0:
            self.model_handler.save_model(self.global_frame_ctr)
        if self.global_frame_ctr % 25000 == 0:
            self.model_handler.save_memory()

        #  It is unlikely that an episode is this long, we're probably stuck somewhere.
        if self.frame_counter % 1500 == 0:
            print('Force restarting game in case of stuck in menu')
            self.init_game(0)
            
        print(status_string)
        if _game_over:
            self.episode += 1
            self.init_game(0)

        pass
