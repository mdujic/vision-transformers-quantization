import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import glob

import shutil
import sys

from joblib import Parallel, delayed

from IPython.display import display

def make_copy(row):
    shutil.copyfile(row.old_image_path, row.image_path)
    return

def crete_image_and_label_dirs(image_dir, label_dir):
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
    return

def coco2yolo(bboxes, height=720, width=1280):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    #bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    # normalizinig
    bboxes[..., 0::2] /= width
    bboxes[..., 1::2] /= height
    # conversion (xmin, ymin) => (xmid, ymid)
    bboxes[..., 0:2] += bboxes[..., 2:4]/2
    return bboxes

def coco2voc(bboxes, height=720, width=1280):
    """
    coco => [xmin, ymin, w, h]
    voc  => [xmin, ymin, xmax, ymax]
    
    """ 
    #bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    # conversion (w, h) => (w, h) 
    bboxes[..., 2:4] += bboxes[..., 0:2]
    return bboxes

def voc2yolo(bboxes, height=720, width=1280):
    """
    voc  => [x1, y1, x2, y1]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    #bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    bboxes[..., 0::2] /= width
    bboxes[..., 1::2] /= height
    bboxes[..., 2] -= bboxes[..., 0]
    bboxes[..., 3] -= bboxes[..., 1]
    bboxes[..., 0] += bboxes[..., 2]/2
    bboxes[..., 1] += bboxes[..., 3]/2
    return bboxes

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_bboxes(img, bboxes, classes, class_ids, colors = None, show_classes = None, bbox_format = 'yolo', class_name = False, line_thickness = 2):  
     
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255 ,0) if colors is None else colors
    if bbox_format == 'yolo':
        for idx in range(len(bboxes)):  
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            if cls in show_classes:
            
                x1 = round(float(bbox[0])*image.shape[1])
                y1 = round(float(bbox[1])*image.shape[0])
                w  = round(float(bbox[2])*image.shape[1]/2) #w/2 
                h  = round(float(bbox[3])*image.shape[0]/2)

                voc_bbox = (x1-w, y1-h, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(get_label(cls)),
                             line_thickness = line_thickness)
    elif bbox_format == 'coco':
        for idx in range(len(bboxes)):  
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            if cls in show_classes:            
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                w  = int(round(bbox[2]))
                h  = int(round(bbox[3]))
                voc_bbox = (x1, y1, x1+w, y1+h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)
    elif bbox_format == 'voc':
        for idx in range(len(bboxes)):  
            bbox  = bboxes[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[cls_id] if type(colors) is list else colors
            if cls in show_classes: 
                x1 = int(round(bbox[0]))
                y1 = int(round(bbox[1]))
                x2 = int(round(bbox[2]))
                y2 = int(round(bbox[3]))
                voc_bbox = (x1, y1, x2, y2)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(cls_id),
                             line_thickness = line_thickness)
    else:
        raise ValueError('wrong bbox format')
    return image

def load_image(image_path):
    return cv2.imread(image_path)[..., ::-1]

def clip_bbox(bboxes_voc, height=720, width=1280):
    """Clip bounding boxes to image boundaries.
    Args:
        bboxes_voc (np.ndarray): bboxes in [xmin, ymin, xmax, ymax] format.
        height (int, optional): height of bbox. Defaults to 720.
        width (int, optional): width of bbox. Defaults to 1280.
    Returns:
        np.ndarray : clipped bboxes in [xmin, ymin, xmax, ymax] format.
    """
    bboxes_voc[..., 0::2] = bboxes_voc[..., 0::2].clip(0, width)
    bboxes_voc[..., 1::2] = bboxes_voc[..., 1::2].clip(0, height)
    return bboxes_voc

def str2annot(data):
    """Generate annotation from string.
    
    Args:
        data (str): string of annotation.
    
    Returns:
        np.ndarray: annotation in array format.
    """
    data  = data.replace('\n', ' ')
    data  = data.strip().split(' ')
    data  = np.array(data)
    annot = data.astype(float).reshape(-1, 5)
    return annot

def annot2str(data):
    """Generate string from annotation.
    
    Args:
        data (np.ndarray): annotation in array format.
    
    Returns:
        str: annotation in string format.
    """
    data   = data.astype(str)
    string = '\n'.join([' '.join(annot) for annot in data])
    return string

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row

def prepare_dataset(main_path, remove_nobbox):
    data_dir = os.path.join(main_path, "data")
    dataset_dir  = os.path.join(data_dir, "tensorflow-great-barrier-reef")
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")

    crete_image_and_label_dirs(image_dir, label_dir)

    df = pd.read_csv(f'{dataset_dir}/train.csv')
    df['old_image_path'] = f'{dataset_dir}/train_images/video_'+df.video_id.astype(str)+'/'+df.video_frame.astype(str)+'.jpg'
    df['image_path']  = f'{image_dir}/'+df.image_id+'.jpg'
    df['label_path']  = f'{label_dir}/'+df.image_id+'.txt'
    df['annotations'] = df['annotations'].progress_apply(eval)

    df['num_bbox'] = df['annotations'].progress_apply(lambda x: len(x))

    if remove_nobbox:
        df = df.query("num_bbox>0")

    image_paths = df.old_image_path.tolist()
    _ = Parallel(n_jobs=-1, backend='threading')(delayed(make_copy)(row) for _, row in tqdm(df.iterrows(), total=len(df)))

    df['bboxes'] = df['annotations'].progress_apply(get_bbox)

    df['width']  = 1280
    df['height'] = 720

    cnt = 0
    all_bboxes = []
    bboxes_info = []
    for row_idx in tqdm(range(df.shape[0])):
        row = df.iloc[row_idx]
        image_height = row.height
        image_width  = row.width
        bboxes_coco  = np.array(row.bboxes).astype(np.float32).copy()
        num_bbox     = len(bboxes_coco)
        names        = ['cots']*num_bbox
        labels       = np.array([0]*num_bbox)[..., None].astype(str)
        ## Create Annotation(YOLO)
        with open(row.label_path, 'w') as f:
            if num_bbox<1:
                annot = ''
                f.write(annot)
                cnt+=1
                continue
            bboxes_voc  = coco2voc(bboxes_coco, image_height, image_width)
            bboxes_voc  = clip_bbox(bboxes_voc, image_height, image_width)
            bboxes_yolo = voc2yolo(bboxes_voc, image_height, image_width).astype(str)
            all_bboxes.extend(bboxes_yolo.astype(float))
            bboxes_info.extend([[row.image_id, row.video_id, row.sequence]]*len(bboxes_yolo))
            annots = np.concatenate([labels, bboxes_yolo], axis=1)
            string = annot2str(annots)
            f.write(string)
    return
