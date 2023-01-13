# -*- coding: utf-8 -*-

import pandas as pd
import json
from glob import glob
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime

def get_time(time_string):
    _time = time_string.strip()
    _time = datetime.strptime(_time, "%M:%S.%f")
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

classis = {
    "response":{
        1 : '정보접수',
        2 : '정보확인',
        3 : '정보탐색',
        4 : '물건전달',
        5 : '동반주행',
    },
    "utterance":{
        1 : '요청',
        2 : '인사',
        3 : '질문',
        4 : '약속',
        5 : '수락',
        6 : '예약'
    },
    "action":{
        1 : '접근',
        2 : '주변파악',
        3 : '정보확인',
        4 : '감정표현',
        5 : '서성거림',
        6 : '자리이탈'
    },
}



def main(args):

    dataset_split_list = args.dataset_split_list
    annotation_json_root = args.annotation_json_root
    transformer_dataset_path = args.transformer_dataset

    df_dataset_list = pd.read_csv(dataset_split_list)
    gb = df_dataset_list.groupby(by='train_val_test')
    trainset = gb.get_group('train')['video_id'].unique()
    validset = gb.get_group('validation')['video_id'].unique()
    testset = gb.get_group('test')['video_id'].unique()

    df_jsons = glob(os.path.join(annotation_json_root, '**/*.json'), recursive=True)
    df_jsons = [x for x in df_jsons if os.path.isfile(x)]
    df_jsons = pd.DataFrame({"json_path":df_jsons})
    df_jsons['video_id'] = df_jsons['json_path'].map(get_key)

    df_annotations = df_jsons
    
    all_qna_data = []
    for i in tqdm(range(len(df_annotations)), desc='JSON Parsing "Robot Response by Human Event"'):

        video_id = df_annotations.loc[i, 'video_id']
        json_path = df_annotations.loc[i, 'json_path']

        train_val_test = None
        if video_id in trainset:
            train_val_test = 'train'
        elif video_id in validset:
            train_val_test = 'validation'
        elif video_id in testset:
            train_val_test = 'test'

        if train_val_test is None:
            continue

        with open(json_path, encoding='utf-8-sig') as f:
            data = json.load(f)

        interactions = data['video']['interactions']

        for interaction in interactions:
            human_event = interaction['human_event']
            actions = human_event['actions']
            utterances = human_event['utterances']
            response = interaction['robot_response']

            human_question = []
            df_actions = pd.DataFrame(actions)
            df_actions.rename(columns={'action_class':'class', 'action_start':'start', 'action_end':'end'}, inplace=True)
            df_actions['type']='action'

            df_utterances = pd.DataFrame(utterances)
            df_utterances.rename(columns={'utterance_intend':'class', 'utterance_start':'start', 'utterance_end':'end', 'utterance_cap':'question'}, inplace=True)
            df_utterances['type']='utterance'

            human_events = pd.concat([df_actions, df_utterances])

            try:
                human_events['start'] = human_events['start'].map(get_time)
                human_events['end'] = human_events['end'].map(get_time)
                human_events.sort_values(by=['start','end', 'type'], inplace=True, ignore_index=True)
                for i in range(len(human_events)):

                    _type = human_events.loc[i, 'type']
                    _class = human_events.loc[i, 'class']
                    _class = eval(_class)
                    _class = list(map(lambda x : classis[_type][x], _class))
                    _class = " ".join(_class)

                    human_question.append(_class)

                    if _type == 'utterance':
                        question = human_events.loc[i, 'question']
                        human_question.append(question)

                human_question = " [SEP] ".join(human_question)

                robot_answer = []
                df_response = pd.DataFrame(response)
                df_response.rename(columns={'action_class':'class'}, inplace=True)
                df_response['type'] = 'response'

                df_response.sort_values(by=['start','end', 'type'], inplace=True, ignore_index=True)

                for i in range(len(df_response)):

                    _type = df_response.loc[i, 'type']
                    _class = df_response.loc[i, 'class']
                    _class = eval(_class)
                    _class = list(map(lambda x : classis[_type][x], _class))
                    _class = " ".join(_class)

                    robot_answer.append(_class)

                    if _type == 'response':
                        answer = df_response.loc[i, 'answer']
                        robot_answer.append(question)

                robot_answer = " [SEP] ".join(robot_answer)   

                all_qna_data.append({
                    'video_id':video_id,
                    'human_event':human_question,
                    'robot_response':robot_answer,
                    'train_val_test':train_val_test
                })

            except ValueError as e:
                pass
            
    df_all_qna_data = pd.DataFrame(all_qna_data)
    df_all_qna_data.to_csv(transformer_dataset_path, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_split_list")
    parser.add_argument("--annotation_json_root", default='./../data/annotations/')
    parser.add_argument("--transformer_dataset", default='transformer_dataset.csv')
    args = parser.parse_args()

    main(args)
