# Author: aqeelanwar
# Created: 27 April,2020, 10:22 PM
# Email: aqeel.anwar@gatech.edu

import argparse
import dlib
from aux_functions import *
from read_cfg import read_cfg
from os.path import join

base_path = os.path.dirname(os.path.abspath(__file__))
mask_types = ["surgical", "surgical_left", "surgical_right", "surgical_green",
              "surgical_green_left", "surgical_green_right", "surgical_blue",
              "surgical_blue_left", "surgical_blue_right", "N95", "N95_right",
              "N95_left", "cloth_left", "cloth_right", "cloth", "KN95", "KN95_left",
              "KN95_right"]


class FaceMaskAugmentor:
    def __init__(self, mask_type="random", pattern="", pattern_weight=0.5, color="", color_weight=0.5, code="", verbose=False, pred_path="dlib_models/shape_predictor_68_face_landmarks.dat"):
        # download_dlib_model()
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

        self.masks = {}

        for mask_type in mask_types:
            print(join(base_path, "masks/masks.cfg"))
            cfg = read_cfg(join(base_path, "masks/masks.cfg"), mask_type)
            self.masks[mask_type] = cv2.imread(join(base_path, cfg.template), cv2.IMREAD_UNCHANGED)


    def _mask_face(self, image, face_location, six_points, angle, type="surgical"):
        debug = False

        # Find the face angle
        threshold = 13
        if angle < -threshold:
            type += "_right"
        elif angle > threshold:
            type += "_left"

        w = image.shape[0]
        h = image.shape[1]
        if not "empty" in type and not "inpaint" in type:
            cfg = read_cfg(config_filename=join(base_path, "masks/masks.cfg"), mask_type=type, verbose=False)
        else:
            if "left" in type:
                str = "surgical_blue_left"
            elif "right" in type:
                str = "surgical_blue_right"
            else:
                str = "surgical_blue"
            cfg = read_cfg(config_filename=join(base_path, "masks/masks.cfg"), mask_type=str, verbose=False)
        img = self.masks[cfg.name]
        # Process the mask if necessary
        if self.args.pattern:
            # Apply pattern to mask
            img = texture_the_mask(img, self.args.pattern, self.args.pattern_weight)

        if self.args.color:
            # Apply color to mask
            img = color_the_mask(img, self.args.color, self.args.color_weight)

        mask_line = np.float32(
            [cfg.mask_a, cfg.mask_b, cfg.mask_c, cfg.mask_f, cfg.mask_e, cfg.mask_d]
        )
        # Warp the mask
        print(cfg.template)
        M, mask = cv2.findHomography(mask_line, six_points)
        dst_mask = cv2.warpPerspective(img, M, (h, w))
        dst_mask_points = cv2.perspectiveTransform(mask_line.reshape(-1, 1, 2), M)
        mask = dst_mask[:, :, 3]
        face_height = face_location[2] - face_location[0]
        face_width = face_location[1] - face_location[3]
        image_face = image[
                     face_location[0] + int(face_height / 2): face_location[2],
                     face_location[3]: face_location[1],
                     :,
                     ]

        image_face = image

        # Adjust Brightness
        mask_brightness = get_avg_brightness(img)
        img_brightness = get_avg_brightness(image_face)
        delta_b = 1 + (img_brightness - mask_brightness) / 255
        dst_mask = change_brightness(dst_mask, delta_b)

        # Adjust Saturation
        mask_saturation = get_avg_saturation(img)
        img_saturation = get_avg_saturation(image_face)
        delta_s = 1 - (img_saturation - mask_saturation) / 255
        dst_mask = change_saturation(dst_mask, delta_s)

        # Apply mask
        mask_inv = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
        img_fg = cv2.bitwise_and(dst_mask, dst_mask, mask=mask)
        out_img = cv2.add(img_bg, img_fg[:, :, 0:3])
        if "empty" in type or "inpaint" in type:
            out_img = img_bg
        # Plot key points

        if "inpaint" in type:
            out_img = cv2.inpaint(out_img, mask, 3, cv2.INPAINT_TELEA)
            # dst_NS = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

        if debug:
            for i in six_points:
                cv2.circle(out_img, (i[0], i[1]), radius=4, color=(0, 0, 255), thickness=-1)

            for i in dst_mask_points:
                cv2.circle(
                    out_img, (i[0][0], i[0][1]), radius=4, color=(0, 255, 0), thickness=-1
                )

        return out_img, mask

    def _mask_image(self, image):
        # Read the image
        original_image = image.copy()
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        args = self.args
        gray = image
        face_locations = args.detector(gray, 1)
        mask_type = args.mask_type
        verbose = args.verbose
        if args.code:
            ind = random.randint(0, len(args.code_count) - 1)
            mask_dict = args.mask_dict_of_dict[ind]
            mask_type = mask_dict["type"]
            args.color = mask_dict["color"]
            args.pattern = mask_dict["texture"]
            args.code_count[ind] += 1

        elif mask_type == "random":
            available_mask_types = get_available_mask_types()
            mask_type = random.choice(available_mask_types)

        if verbose:
            tqdm.write("Faces found: {:2d}".format(len(face_locations)))
        # Process each face in the image
        masked_images = []
        mask_binary_array = []
        mask = []
        for (i, face_location) in enumerate(face_locations):
            shape = args.predictor(gray, face_location)
            shape = face_utils.shape_to_np(shape)
            face_landmarks = shape_to_landmarks(shape)
            face_location = rect_to_bb(face_location)
            # draw_landmarks(face_landmarks, image)
            six_points_on_face, angle = get_six_points(face_landmarks, image)
            mask = []
            if mask_type != "all":
                if len(masked_images) > 0:
                    image = masked_images.pop(0)
                image, mask_binary = self._mask_face(
                    image, face_location, six_points_on_face, angle, type=mask_type
                )

                # compress to face tight
                face_height = face_location[2] - face_location[0]
                face_width = face_location[1] - face_location[3]
                masked_images.append(image)
                mask_binary_array.append(mask_binary)
                mask.append(mask_type)
            else:
                available_mask_types = get_available_mask_types()
                for m in range(len(available_mask_types)):
                    if len(masked_images) == len(available_mask_types):
                        image = masked_images.pop(m)
                    img, mask_binary = self._mask_face(
                        image,
                        face_location,
                        six_points_on_face,
                        angle,
                        args,
                        type=available_mask_types[m],
                    )
                    masked_images.insert(m, img)
                    mask_binary_array.insert(m, mask_binary)
                mask = available_mask_types
                cc = 1

        return masked_images, mask, mask_binary_array


    def __call__(self, img):
        return self._mask_image(img)
