# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
import dlib
from aux_functions import *


class FaceMaskAugmentor:
    def __init__(self, mask_type="random", pattern="", pattern_weight=0.5, color="#0473e2", color_weight=0.5, code=""):
        self.args = argparse.Namespace(
            mask_type=mask_type,
            pattern=pattern,
            pattern_weight=pattern_weight,
            color=color,
            color_weight=color_weight,
            code=code
    )

    def __call__(self, img):
        return mask_image(img, args)
