import os
import urllib.request
from tqdm import tqdm
from PIL import Image
import mediapipe
from mediapipe.framework.formats import landmark_pb2
from controlnet_aux import OpenposeDetector
import numpy as np
import copy
import json


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

base_path = os.path.dirname(__file__)
mp_model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'


right_style_lm = copy.deepcopy(mediapipe.solutions.drawing_styles.get_default_hand_landmarks_style())
left_style_lm = copy.deepcopy(mediapipe.solutions.drawing_styles.get_default_hand_landmarks_style())
right_style_lm[0].color=(251, 206, 177)
left_style_lm[0].color=(255, 255, 225)

openpose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


def crop_zero_regions(arr):
    while len(arr.shape) > 2:
        arr = np.sum(arr, axis=-1)
    
    colsum = np.sum(arr, axis=1) != 0
    minh = np.argmax(colsum)
    maxh = colsum.shape[0] - 1 - np.argmax(colsum[::-1])

    rowsum = np.sum(arr, axis=0) != 0
    minw = np.argmax(rowsum)
    maxw = rowsum.shape[0] - 1 - np.argmax(rowsum[::-1])
    return minh, maxh, minw, maxw

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.zeros_like(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        # Draw the hand landmarks.

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        if handedness[0].category_name == "Left":
            mediapipe.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mediapipe.solutions.hands.HAND_CONNECTIONS,
                    left_style_lm,
                    mediapipe.solutions.drawing_styles.get_default_hand_connections_style())
        if handedness[0].category_name == "Right":
            mediapipe.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mediapipe.solutions.hands.HAND_CONNECTIONS,
                    right_style_lm,
                    mediapipe.solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image

def mediepipe_preprocess_image(img):
    """img(input): numpy array
       annotated_image(output): numpy array
    """
    # STEP 2: Create an HandLandmarker object.
    base_options = mediapipe.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mediapipe.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    detector = mediapipe.tasks.vision.HandLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=np.array(img))

    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    return annotated_image

def run(data_in, data_out, res, indexed=True):

    mp_model = os.path.join(base_path, 'hand_landmarker.task')
    if not os.path.exists(mp_model):
        download_url(mp_model_url, mp_model)
    if not os.path.exists(mp_model):
        raise ValueError("Cannot download mediepipe model!")

    if not os.path.exists(data_in):
        raise ValueError("Cannot find data!")

    os.makedirs(data_out, exist_ok=True)

    idx = 0
    metadata = []

    for img_file in os.listdir(data_in):
        try:
            img = Image.open(os.path.join(data_in, img_file))
        except:
            continue

        if img.size[0] > img.size[1]:
            resize = (min(res, img.size[0]), int((img.size[1]/img.size[0]) * min(res, img.size[0])))
        else:
            resize = (int((img.size[0]/img.size[1]) * min(res, img.size[1])), min(res, img.size[1]))        
        img = img.resize(resize, Image.Resampling.BICUBIC)

        out = mediepipe_preprocess_image(img).astype(np.int32)
        out_pose = np.array(openpose_model(img).resize(resize, Image.Resampling.NEAREST), dtype=np.int32)

        if out.sum() == 0 or out_pose.sum() == 0:
            continue

        cond_images = np.concatenate((out, out_pose), axis=2)

        minh, maxh, minw, maxw = crop_zero_regions(cond_images)
        cond_center = (((minh+maxh)/2), ((minw+maxw)/2))
        img_center = ((img.size[1]/2), (img.size[0]/2))
        offset_normalized = (((cond_center[0]-img_center[0])/img.size[1]), ((cond_center[1]-img_center[1])/img.size[0]))
        cropped_image = cond_images[minh:maxh+1, minw:maxw+1]

        image_name = os.path.splitext(img_file)[0] if not indexed else str(idx)
        image_pose_name = f'{image_name}_pose'

        Image.fromarray(cropped_image[:, :, :3].astype(np.uint8)).save(os.path.join(data_out, f'{image_name}.png'))
        Image.fromarray(cropped_image[:, :, 3:].astype(np.uint8)).save(os.path.join(data_out, f'{image_pose_name}.png'))

        idx = idx + 1
        metadata.append({
            'name': f'{image_name}.png',
            'name_pose': f'{image_pose_name}.png',
            'orig_name': img_file,
            'offset': offset_normalized,
            'size_hw': [img.size[1], img.size[0]]
        })

    with open(os.path.join(data_out, "meta.json"), "w") as outfile:
        json.dump(metadata, outfile, indent=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data_in', type=str)
    parser.add_argument('--data_out', type=str, default='tup_dataset')
    parser.add_argument('--res', type=int, default=512)
    args = parser.parse_args()

    run(args.data_in, args.data_out, args.res)
