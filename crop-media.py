import face_alignment
import skimage.io
import numpy as np
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os
import imageio
import warnings
warnings.filterwarnings("ignore")

def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yA - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)

def compute_bbox_image(bbox, frame_shape, image_shape, increase_area=0.1):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    # Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    return left, top, right, bot

def process_image(args):
    device = 'cpu' if args.cpu else 'cuda'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    frame = skimage.io.imread(args.inp)
    frame_shape = frame.shape

    bboxes = extract_bbox(frame, fa)
    if len(bboxes) == 0:
        print("No faces detected.")
        return

    for bbox in bboxes:
        left, top, right, bot = compute_bbox_image(bbox, frame_shape, args.image_shape, args.increase)
        crop = frame[top:bot, left:right]
        if args.image_shape is not None:
            crop = img_as_ubyte(resize(crop, args.image_shape, anti_aliasing=True))
        output_path = os.path.join(args.out_folder, os.path.basename(args.inp))
        skimage.io.imsave(output_path, crop)
        print(f"Cropped image saved to {output_path}")

def process_video(args):
    device = 'cpu' if args.cpu else 'cuda'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    reader = imageio.get_reader(args.inp)
    fps = reader.get_meta_data()['fps']
    frames = []
    for i, frame in enumerate(reader):
        bboxes = extract_bbox(frame, fa)
        if len(bboxes) == 0:
            continue
        for bbox in bboxes:
            left, top, right, bot = compute_bbox_image(bbox, frame.shape, args.image_shape, args.increase)
            crop = frame[top:bot, left:right]
            if args.image_shape is not None:
                crop = img_as_ubyte(resize(crop, args.image_shape, anti_aliasing=True))
            frames.append(crop)
    output_path = os.path.join(args.out_folder, os.path.basename(args.inp).replace('.mp4', '.gif'))
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Cropped video saved to {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with initial bbox")
    parser.add_argument("--inp", required=True, help='Input image or video')
    parser.add_argument("--out_folder", required=True, help='Output folder')
    parser.add_argument("--min_frames", type=int, default=150, help='Minimum number of frames')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="CPU mode.")
    parser.add_argument("--media_type", choices=['image', 'video'], required=True, help="Type of media to process")

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    if args.media_type == 'image':
        process_image(args)
    else:
        process_video(args)
