import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from pathlib import Path
from tqdm.auto import tqdm
import argparse
import os
from PIL import Image
import csv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./data/data_object_image_2')
    parser.add_argument('--output_dir', type=str, default='labels_clean')
    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)
    args.label_path = args.base_dir / 'training' / 'label_2'
    args.img_path = args.base_dir / 'training' / 'image_2'
    args.calib_path = args.base_dir / 'training' / 'calib'
    args.ims = sorted(list(args.img_path.glob('*')))
    args.labels = sorted(list(args.label_path.glob('*')))
    assert len(args.ims) == len(args.labels)
    assert len(args.ims) > 0

    args.label_colors = {
        'Car': (255, 0, 0),
        'Van': (255, 255, 0),
        'Truck': (255, 255, 255),
        'Pedestrian': (0, 255, 255),
        'Person_sitting': (0, 255, 255),
        'Cyclist': (0, 128, 255),
        'Tram': (128, 0, 0),
        'Misc': (0, 255, 255),
        'DontCare': (255, 255, 0)
    }
    args.label_cols = [
        'label', 'truncated', 'occluded', 'alpha',
        'bbox_xmin', 'bbox_ymin', 'bbox_xmax',
        'bbox_ymax', 'dim_height', 'dim_width', 'dim_length',
        'loc_x', 'loc_y', 'loc_z', 'rotation_y', 'score'
    ]

    args.df = pd.DataFrame({
        'image': args.ims,
        'label': args.labels
    })

    return args


def check_data(args):
    print(pd.read_csv(args.df['label'][55], sep=" ", names=args.label_cols[:15], usecols=args.label_cols[:15]))


def open_image(p):
    im = cv.imread(str(p))
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


def draw_box2d(args, idx, ax=None):
    sample = args.df.iloc[idx, :]
    img = open_image(sample['image'])
    labels = pd.read_csv(sample['label'], sep=" ", names=args.label_cols[:15], usecols=args.label_cols[:15])
    for index, row in labels.iterrows():
        left_corner = (int(row.bbox_xmin), int(row.bbox_ymin))
        right_corner = (int(row.bbox_xmax), int(row.bbox_ymax))

        if row.label == 'DontCare':
            continue

        label_color = args.label_colors.get(row.label, (0, 255, 0))
        img = cv.rectangle(img, left_corner, right_corner, label_color, 2)
        img = cv.putText(img, row.label,
                         (left_corner[0] + 10, left_corner[1] - 4),
                         cv.FONT_HERSHEY_SIMPLEX, 1,
                         label_color, 3)

    if ax is None:
        plt.imsave('test.png', img)
    else:
        ax.savefig('test.png')


def generate_kitti_4_yolo(args):
    glob_num_total = 0
    glob_num_filtered = 0
    glob_removed_by = {'class': 0, 'height': 0, 'truncation': 0, 'occlusion': 0}

    all_classes = []

    OUT_LABELS_DIR = args.output_dir

    IGNORE_CLASSES = ['DontCare', 'Tram', 'Misc', 'Truck']

    class_names = ['Car', 'Pedestrian', 'Van', 'Cyclist', 'Truck', 'Misc', 'Tram', 'Person_sitting', 'DontCare']

    CLASS_NUMBERS = {
        name: idx for idx, name in enumerate(class_names)
    }

    def getSampleId(path):
        basename = os.path.basename(path)
        return os.path.splitext(basename)[0]

    def get_class_number(class_name):
        if class_name not in IGNORE_CLASSES:
            return CLASS_NUMBERS[class_name]

    def convertToYoloBBox(bbox, size):
        # Yolo uses bounding bbox coordinates and size relative to the image size.
        # This is taken from https://pjreddie.com/media/files/voc_label.py .
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (bbox[0] + bbox[1]) / 2.0
        y = (bbox[2] + bbox[3]) / 2.0
        w = bbox[1] - bbox[0]
        h = bbox[3] - bbox[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def readRealImageSize(img_path):
        # This loads the whole sample image and returns its size.
        return Image.open(img_path).size

    def parseSample(lbl_path, img_path):
        num_total = 0
        num_filtered = 0
        removed_by = {'class': 0, 'height': 0, 'truncation': 0, 'occlusion': 0}
        with open(lbl_path) as csv_file:
            reader = csv.DictReader(
                csv_file,
                fieldnames=[
                    "type",
                    "truncated",
                    "occluded",
                    "alpha",
                    "bbox2_left",
                    "bbox2_top",
                    "bbox2_right",
                    "bbox2_bottom",
                    "bbox3_height",
                    "bbox3_width",
                    "bbox3_length",
                    "bbox3_x",
                    "bbox3_y",
                    "bbox3_z",
                    "bbox3_yaw",
                    "score"],
                delimiter=" ")
            yolo_labels = []
            for row in reader:
                num_total += 1
                all_classes.append(row['type'])
                class_number = get_class_number(row["type"])
                if class_number is None:
                    removed_by['class'] += 1
                    continue
                size = readRealImageSize(img_path)
                # size = readFixedImageSize()
                # Image coordinate is in the top left corner.
                bbox = (
                    float(row["bbox2_left"]),
                    float(row["bbox2_right"]),
                    float(row["bbox2_top"]),
                    float(row["bbox2_bottom"])
                )
                # remove if more difficult than hard (height >= 25 and truncation <= 0.5 and occlusion <= 2)
                height = bbox[3] - bbox[2]
                if height < 25:
                    removed_by['height'] += 1
                    continue
                if float(row['truncated']) > 0.5:
                    removed_by['truncation'] += 1
                    continue
                if float(row['occluded']) > 2:
                    removed_by['occlusion'] += 1
                    continue
                yolo_bbox = convertToYoloBBox(bbox, size)
                # Yolo expects the labels in the form:
                # <object-class> <x> <y> <width> <height>.
                yolo_label = (class_number,) + yolo_bbox
                yolo_labels.append(yolo_label)
                num_filtered += 1
        return yolo_labels, num_total, num_filtered, removed_by

    args = argparse.Namespace(
        label_dir=str(args.label_path),
        image_2_dir=str(args.img_path),
    )

    if not os.path.exists(OUT_LABELS_DIR):
        os.makedirs(OUT_LABELS_DIR)

    print("Generating darknet labels...")
    sample_img_paths = []

    for dir_path, _, files in os.walk(args.label_dir):
        with tqdm(total=len(files)) as pbar:
            for file_name in files:
                if not file_name.endswith(".txt"):
                    continue

                # get paths
                lbl_path = os.path.join(dir_path, file_name)
                sample_id = getSampleId(lbl_path)
                img_path = os.path.join(args.image_2_dir, f"{sample_id}.png")
                sample_img_paths.append(img_path)

                # parse labels
                yolo_labels, num_total, num_filtered, removed_by = parseSample(lbl_path, img_path)

                # statistics
                glob_num_total += num_total
                glob_num_filtered += num_filtered
                for key in removed_by:
                    glob_removed_by[key] += removed_by[key]

                # write labels
                with open(os.path.join(OUT_LABELS_DIR, f"{sample_id}.txt"), "w") as yolo_label_file:
                    for lbl in yolo_labels:
                        yolo_label_file.write("{} {} {} {} {}\n".format(*lbl))
                pbar.update(1)
                pbar.set_postfix_str(f'num_total: {glob_num_total}, num_filtered: {glob_num_filtered} ({glob_num_filtered / glob_num_total:.2%}), removed_by: {glob_removed_by}')
    print(f'Written {len(sample_img_paths)} labels to {OUT_LABELS_DIR}.')
    import json
    with open('kitti_classes.json', 'w') as f:
        json.dump(CLASS_NUMBERS, f)
    print(CLASS_NUMBERS)
    print(f'All classes: {set(all_classes)}')
    print(f'Glob num total: {glob_num_total}')
    print(f'Glob num filtered: {glob_num_filtered} ({glob_num_filtered / glob_num_total:.2%})')
    print(f'Glob removed by:\n{glob_removed_by}')
    # percentage of removed samples by height, truncation and occlusion
    print(f'Glob removed by height: {glob_removed_by["height"] / glob_num_total:.2%}')
    print(f'Glob removed by truncation: {glob_removed_by["truncation"] / glob_num_total:.2%}')
    print(f'Glob removed by occlusion: {glob_removed_by["occlusion"] / glob_num_total:.2%}')
    print(f'Glob removed by class: {glob_removed_by["class"] / glob_num_total:.2%}')
    print(f'Glob removed by height, truncation and occlusion: {(glob_removed_by["height"] + glob_removed_by["truncation"] + glob_removed_by["occlusion"]) / glob_num_total:.2%}')


def main(args):
    check_data(args)
    draw_box2d(args, 55)
    print(' === STARTING GENERATION === ')
    generate_kitti_4_yolo(args)


if __name__ == '__main__':
    args = get_args()
    main(args)
