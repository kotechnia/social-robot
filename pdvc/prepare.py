# -*- coding: utf-8 -*-

from datetime import datetime
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np
import json
import yaml
import os

#class_mapper = {
#    "human_action" : {
#        1 : "고객 행위 의도 접근",
#        2 : "고객 행위 의도 주변파악",
#        3 : "고객 행위 의도 정보확인",
#        4 : "고객 행위 의도 감정표현",
#        5 : "고객 행위 의도 서성거림",
#        6 : "고객 행위 의도 자리이탈",
#    },
#}

def get_time(time_string):
    _time = time_string.strip()
    _time = datetime.strptime(_time, "%M:%S.%f")
    _time = _time.minute * 60. + _time.second * 1. + _time.microsecond * 1e-6
    return _time

def get_key(path):
    try:
        path = os.path.basename(path)
        key, _ = os.path.splitext(path)
        key = key.strip()
#    key = key.split('_')
#    if key[-1] in ['color', 'rgb']:
#        key = key[:-1]
#    key = "_".join(key)
#    key = key.strip()
    except: 
        print(path)
    return key


def trunc(_time):
    _time = np.trunc(_time*10.) / 10.
    return _time

def refine_data_pdvc(data):
    data = data['video']
    video_id = get_key(data['video_id'])
#    video_id = data['video_file_name'].replace('/mp4', '')
    video_duration = data['video_duration']
    _data = {'duration':video_duration, 'timestamps':[], 'sentences':[]}
    
    for interaction in data['interactions']:
        for action in interaction['human_event']['actions']:
#            action_class = eval(action['action_class'])
#            action_class = list(map(lambda x: class_mapper['human_action'][x], action_class))
            timestamp_start = action['action_start']
            timestamp_end = action['action_end']
            
            try:
                timestamp_start = trunc(get_time(timestamp_start))
                timestamp_end = trunc(get_time(timestamp_end))
                
            
                if timestamp_start < timestamp_end: 
                    _data['timestamps'].append([timestamp_start, timestamp_end])
                    _data['sentences'].append(action['action_discription'])
                    
                else:
                    #print(f'ERROR : Time Reverse, start:{timestamp_start}, end:{timestamp_end}' )
                    pass
            except Exception as e:
                #print(e)
                pass
            
    if len(_data['timestamps']) == 0:
        return None, None
    
    return _data, video_id

def main(args):
    dataset_split_list = args.dataset_split_list
    video_root = args.video_root
    json_root = args.json_root
    feature_root = args.feature_root
    base_cfg_path = args.base_cfg_path
    cfg_path = args.cfg_path
    pdvc_dataset_path = args.pdvc_dataset_path
    
    df_dataset_list = pd.read_csv(dataset_split_list)

    df_videos = glob(os.path.join(feature_root, '**/**/*.npy'), recursive=True)
    df_videos = pd.DataFrame({"video_path":list(set(df_videos))})
    df_videos['video_id'] = df_videos['video_path'].map(get_key)

    df_jsons = glob(os.path.join(json_root, '**/**/**/*.json'), recursive=True)
    df_jsons = [x for x in df_jsons if os.path.isfile(x)]
    df_jsons = pd.DataFrame({"json_path":list(set(df_jsons))})
    df_jsons['video_id'] = df_jsons['json_path'].map(get_key)

    df_annotations = pd.merge(df_videos, df_jsons, on=['video_id'])

    all_caption_data = {}
    for i in tqdm(range(len(df_annotations)), desc='Parsing json'):

        json_path = df_annotations.loc[i, 'json_path']
        with open(json_path, encoding='utf-8-sig') as f:
            data = json.load(f)

        data, video_id = refine_data_pdvc(data)
        if video_id is None : continue


        all_caption_data[video_id] = data

#    video_id_mapper = list(all_caption_data.keys())   
#    video_id_mapper = {get_key(id_) : id_ for id_ in video_id_mapper}   

    datasets = {'train':{}, 'validation':{}, 'test':{}}
    datasets_gt = {'train':{}, 'validation':{},'test':{}}

    for tvt in ['train', 'validation', 'test']:
        df_dataset = df_dataset_list[df_dataset_list['train_val_test'] == tvt].reset_index(drop=True)
        for i in tqdm(range(len(df_dataset)), desc=f"{tvt} split"):
            video_id = df_dataset.loc[i, 'video_id']

            try:
#                video_id = video_id_mapper[video_id]    
                datasets[tvt][video_id] = all_caption_data[video_id]
                datasets_gt[tvt][video_id] = '  '.join(all_caption_data[video_id]['sentences'])
            except Exception as e:
#                print(e)
#                print(video_id)
                pass

    cnt = len(datasets['train'])+len(datasets['validation'])+len(datasets['test'])
    print(f"total dataset {len(datasets['train'])} {len(datasets['validation'])} {len(datasets['test'])}")
    print(f"ratio dataset {len(datasets['train'])/cnt} {len(datasets['validation'])/cnt}, {len(datasets['test'])/cnt}")

    pdvc_dataset_name, ext = os.path.splitext(pdvc_dataset_path)
    pdvc_dataset_path = pdvc_dataset_name+'_{}'+ext
    pdvc_dataset_gt_path = pdvc_dataset_name+'_gt_{}'+ext
    pdvc_dataset_path, pdvc_dataset_gt_path

    for mode in ['train', 'validation', 'test']:
        json.dump(datasets[mode], open(pdvc_dataset_path.format(mode), 'w'), indent=4, ensure_ascii=False)
        json.dump(datasets_gt[mode], open(pdvc_dataset_gt_path.format(mode), 'w'), indent=4, ensure_ascii=False)

    try:
        with open(base_cfg_path) as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
    except:
        configs={}

    model_id = os.path.basename(cfg_path)
    model_id, _  = os.path.splitext(model_id)
    configs['id'] = model_id
    configs['train_caption_file'] = pdvc_dataset_path.format('train')
    configs['val_caption_file'] = pdvc_dataset_path.format('validation')
    configs['eval_caption_file'] = pdvc_dataset_path.format('test')
    configs['eval_caption_file_para'] = pdvc_dataset_gt_path.format('test')
    configs['gt_file_for_eval'] = [pdvc_dataset_path.format('validation')]
    configs['gt_file_for_para_eval'] = [pdvc_dataset_gt_path.format('validation')]
    if feature_root is not None:
        configs['visual_feature_folder'] = [feature_root]

    with open(cfg_path, 'w') as f:
        yaml.dump(configs, f)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_split_list',default='../dataset_split_list.csv', type=str)
    parser.add_argument('--video_root', default='../data/videos', type=str)
    parser.add_argument('--json_root', default='../data/annotations', type=str)
    parser.add_argument('--feature_root', default=None)
    parser.add_argument('--base_cfg_path', default='../cfgs/social_robot.yml', type=str)
    parser.add_argument('--cfg_path', default='../cfgs/social_robot.yml', type=str)
    parser.add_argument('--pdvc_dataset_path', default='data/videos', type=str)
    
    args = parser.parse_args()
    main(args)
