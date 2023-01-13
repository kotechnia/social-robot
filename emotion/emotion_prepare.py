# -*- coding: utf-8 -*-

import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime
import cv2

def classnum_to_class(num):
    if num == 1:
        return '기쁨'
    elif num == 2:
        return '화남'
    elif num == 3:
        return '놀람'
    elif num == 4:
        return '무표정'
    elif num == 5:
        return '모름'
    else:
        return '모름'


def get_time(time_string):
    _time = time_string.strip()
    _time = datetime.strptime(_time, "%S.%f")
    _time = _time.minute * 60. + _time.second * 1. + _time.microsecond * 1e-6
    return _time

def get_key(path):
    path = os.path.basename(path)
    key, _ = os.path.splitext(path)
    key = key.strip()
    key = key.split('_')
    if key[-1] in ['color', 'rgb']:
        key = key[:-1]
    key = "_".join(key)
    key = key.strip()
    return key


def main(args):
    
    dataset_split_list = args.dataset_split_list
    video_root = args.video_root
    json_root = args.annotation_root
    image_root = args.image_root
    extract_image = args.extract_image
    dataset_path = args.dataset_path

    df_dataset_list = pd.read_csv(dataset_split_list)

    df_videos = glob(os.path.join(video_root, '**/*.mp4'), recursive=True)
    df_videos = pd.DataFrame({"video_path":df_videos})
    df_videos['video_id'] = df_videos['video_path'].map(get_key)

    df_jsons = glob(os.path.join(json_root, '**/*.json'), recursive=True)
    df_jsons = [x for x in df_jsons if os.path.isfile(x)]
    df_jsons = pd.DataFrame({"json_path":df_jsons})
    df_jsons['video_id'] = df_jsons['json_path'].map(get_key)

    df_annotations = pd.merge(df_videos, df_jsons, on=['video_id'])

    all_face_data = {}
    for i in tqdm(range(len(df_annotations)), desc='Parsing json for face classification'):

        video_id = df_annotations.loc[i, 'video_id']
        video_path = df_annotations.loc[i, 'video_path']
        json_path = df_annotations.loc[i, 'json_path']

        with open(json_path, encoding='utf-8-sig') as f:
            data = json.load(f)

        interactions = data['video']['interactions']

        for interaction in interactions:
            faces = interaction['faces']

            for face in faces:
                if face['face_bbox'] is None:
                    continue
                else:
                    try:
                        _time = get_time(face['time'])
                        face_class = face['face_class']
                        face_bbox = face['face_bbox']
                    except KeyError as e:
                        #tqdm.write(f"{json_path} : {e}")
                        continue

                    try:
                        all_face_data[video_id]['face_info'].append(
                            {'time': _time, 'face_class': face_class, 'face_bbox': face_bbox})
                    except KeyError as e:
                        all_face_data[video_id] = {}
                        all_face_data[video_id]['video_path'] = video_path
                        all_face_data[video_id]['face_info'] = [
                            {'time': _time, 'face_class': face_class, 'face_bbox': face_bbox}]     

    face_datasets = []

    for video_id in tqdm(all_face_data.keys(), desc='Extract faces from video'):
        data = all_face_data[video_id]
        video_path = data['video_path']
        face_infos = data['face_info']

        if extract_image:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

        for face_info in tqdm(face_infos, leave=False, desc='Extract faces from image'):
            time = face_info['time']

            if extract_image:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * time))
                ret, frame = cap.read()
            else:
                ret = True

            if ret: 
                face_class = face_info['face_class']
                face_bbox = face_info['face_bbox']

                for face_id in face_class.keys():
                    x, y, w, h = face_bbox[face_id]
                    _class = eval(face_class[face_id])[0]

                    filename = f'{video_id}_{time:06.2f}_{face_id}_{_class}.png'

                    filepath = os.path.join(image_root, video_id, filename)

                    if os.path.isfile(filepath):
                        pass
                    else:
                        if extract_image:
                            os.makedirs(os.path.join(image_root, video_id), exist_ok=True)
                            roi = frame[int(y):int(y+h), int(x):int(x+w)] 
                            cv2.imwrite(filepath, roi)

                    face_datasets.append({
                        'image_path':filepath, 
                        'image_name':filename,
                        'class':_class, 
                        'class_name':classnum_to_class(_class), 
                        'video_id':video_id
                    })

        if extract_image:
            cap.release()

    df_face_datasets = pd.DataFrame(face_datasets)
    df_face_datasets = pd.merge(df_face_datasets, df_dataset_list, on=['video_id'])
    isfile = df_face_datasets['image_path'].map(os.path.isfile)
    df_face_datasets = df_face_datasets.loc[isfile]
    df_face_datasets.to_csv(dataset_path, index=False)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_split_list", default='../dataset_split_list.csv')
    parser.add_argument("--video_root", default='../data/videos')
    parser.add_argument("--annotation_root", default='../data/annotations')
    parser.add_argument("--image_root", default='../data/faces')
    parser.add_argument('--extract_image', action='store_true')
    parser.add_argument("--dataset_path", default='emotion_dataset.csv')
    args = parser.parse_args()

    main(args)
