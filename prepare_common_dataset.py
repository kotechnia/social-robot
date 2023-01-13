from glob import glob
import pandas as pd
import numpy as np
import os


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
    video_root = args.video_root
    json_root = args.annotation_root
    dataset_split_list = args.dataset_split_list
                
    df_videos = glob(os.path.join(video_root, '**/*.mp4'), recursive=True)
    df_videos = pd.DataFrame({"video_path":df_videos})
    df_videos['video_id'] = df_videos['video_path'].map(get_key)
                        
    df_jsons = glob(os.path.join(json_root, '**/*.json'), recursive=True)
    df_jsons = [x for x in df_jsons if os.path.isfile(x)]
    df_jsons = pd.DataFrame({"json_path":df_jsons})
    df_jsons['video_id'] = df_jsons['json_path'].map(get_key)
                            
    df_annotations = pd.merge(df_videos, df_jsons, on=['video_id'])
    df_annotations = df_annotations[['video_id']]
    indices = list(df_annotations.index)
    np.random.shuffle(indices)
    train_indices = indices[:int(len(indices)*0.8)]
    valid_indices = indices[int(len(indices)*0.8):int(len(indices)*0.9]
    test_indices = indices[int(len(indices)*0.9):]
    
    df_annotations['train_val_test']='train'
    df_annotations.loc[train_indices, 'train_val_test']='train'
    df_annotations.loc[valid_indices, 'train_val_test']='validation'
    df_annotations.loc[test_indices, 'train_val_test']='test'
    df_annotations.to_csv(dataset_split_list, index=False)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", default='data/videos')
    parser.add_argument("--annotation_root", default='data/annotations')
    parser.add_argument("--dataset_split_list", default='dataset_split_list.csv')
    args = parser.parse_args()
    main(args)
