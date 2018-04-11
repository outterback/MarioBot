from serpent.game import Game
from pathlib import Path
from .api.api import MarioBros1API
import time
from serpent.utilities import Singleton




class SerpentMarioBros1Game(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"


        rom_path = Path("/home/oscar/Games/NES/super_mario_bros_1.nes")
        kwargs["window_name"] = "FCEUX 2.2.3 debug"
        kwargs["executable_path"] = "fceux " + str(rom_path)


        super().__init__(**kwargs)

        self.api_class = MarioBros1API
        self.api_instance = None

    def after_launch(self):
        self.is_launched = True

        time.sleep(5)
        windows = self.window_controller.locate_window(self.window_name).split("\n")
        print(f'Windows: {windows}')
        self.window_id = self.window_controller.locate_window(self.window_name).split("\n")[0]

        #self.window_controller.resize_window(self.window_id, 1024, 768)  #Optional - Can change or skip

        self.window_controller.move_window(self.window_id, 0, 0)
        self.window_controller.focus_window(self.window_id)

        self.window_geometry = self.extract_window_geometry()
        print(self.window_geometry)


    @property
    def screen_regions(self):
        regions = {
            "SAMPLE_REGION": (0, 0, 0, 0)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets
