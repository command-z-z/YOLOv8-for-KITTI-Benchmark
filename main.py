from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm.auto import tqdm
import shutil
from PIL import Image
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='./data/data_object_image_2/')
    parser.add_argument('--label_path', type=str, default='labels_clean')
    parser.add_argument('--kitti_classes', type=str, default='kitti_classes.json')
    parser.add_argument('--train_split_path', type=str, default='kitti_splits/train_split.txt')
    parser.add_argument('--val_split_path', type=str, default='kitti_splits/val_split.txt')
    parser.add_argument('--type', type=str, default='train', choices=['preprocess', 'train', 'evaluate', 'predict', 'export'])
    parser.add_argument('--weight', type=str, default='./yolov8x-kitti-clean/train/weights/best.pt')
    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)
    args.img_path = args.base_dir / 'training' / 'image_2'
    args.test_path = args.base_dir / 'testing' / 'image_2'
    args.label_path = Path(args.label_path)
    with open(args.kitti_classes,'r') as f:
        args.classes = json.load(f)
    args.ims = sorted(list(args.img_path.glob('*')))
    args.test_ims = sorted(list(args.test_path.glob('*')))
    args.labels = sorted(list(args.label_path.glob('*')))
    assert len(args.ims) == len(args.labels)
    assert len(args.ims) > 0
    
    with open(args.train_split_path, 'r') as f:
        train_sequences = [x.strip() for x in f.readlines()]
    with open(args.val_split_path, 'r') as f:
        val_sequences = [x.strip() for x in f.readlines()]

    pairs = list(zip(args.ims, args.labels))
    args.train_pairs = [x for x in pairs if x[0].stem in train_sequences]
    args.val_pairs = [x for x in pairs if x[0].stem in val_sequences]

    print(f"Found {len(args.train_pairs)} training pairs and {len(args.val_pairs)} val pairs")

    return args

def train():
    model = YOLO('yolov8x.yaml')
    model = YOLO('yolov8x.pt')
    _ = model.train(
        data='kitti.yaml', 
        epochs=50,
        patience=50,
        mixup=0.1,
        project='yolov8x-kitti-clean',
        device=0,
        batch=9,
    )


def evaluate():
    model = YOLO(args.weight)
    _ = model.val()

def predict():
    model = YOLO(args.weight)
    img_paths_chunks = np.array_split(args.test_ims, len(args.test_ims) // 16)
    for img_paths in tqdm(img_paths_chunks):
        _ = model.predict(img_paths.tolist(), conf=0.5, save=True, save_conf=True, save_txt=True)

def export():
    model = YOLO(args.weight)
    model.export(format='engine')


def preprocess(args):
    train_path = Path('train').resolve()
    train_path.mkdir(exist_ok=True)
    valid_path = Path('valid').resolve()
    valid_path.mkdir(exist_ok=True)

    print(f'Copying {len(args.train_pairs)} training pairs')
    for t_img, t_lb in tqdm(args.train_pairs):
        im_path = train_path / t_img.name
        lb_path = train_path / t_lb.name
        shutil.copy(t_img,im_path)
        shutil.copy(t_lb,lb_path)

    print(f'Copying {len(args.val_pairs)} validation pairs')
    for t_img, t_lb in tqdm(args.val_pairs):
        im_path = valid_path / t_img.name
        lb_path = valid_path / t_lb.name
        shutil.copy(t_img,im_path)
        shutil.copy(t_lb,lb_path)

    yaml_file = 'names:\n' + '\n'.join(f'- {c}' for c in args.classes)
    yaml_file += f'\nnc: {len(args.classes)}'
    yaml_file += f'\ntrain: {str(train_path)}\nval: {str(valid_path)}'
    with open('kitti.yaml','w') as f:
        f.write(yaml_file)

if __name__ == '__main__':
    args = get_args()
    if args.type == "preprocess":
        preprocess(args)
    elif args.type == "train":
        train()
    elif args.type == "evaluate":
        evaluate()
    elif args.type == "predict":
        predict()
    elif args.type == "export":
        export()
