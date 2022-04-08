# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
import dlib
from aux_functions import *


class FaceMaskAugmentor:
    def __init__(self, mask_type="random", pattern="", pattern_weight=0.5, color="#0473e2", color_weight=0.5, code="", verbose=False, pred_path="dlib_models/shape_predictor_68_face_landmarks.dat"):
        download_dlib_model()
        self.args = argparse.Namespace(
            mask_type=mask_type,
            pattern=pattern,
            pattern_weight=pattern_weight,
            verbose=verbose,
            color=color,
            color_weight=color_weight,
            code=code,
            predictor=dlib.shape_predictor(pred_path),
            detector=dlib.get_frontal_face_detector()
         )
    def __call__(self, img):
        return mask_image(img, self.args)
