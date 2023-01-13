import tensorflow as tf
from glob import glob
import json
from datetime import datetime
from tqdm import tqdm
from tokenizer import Tokenizer
import pandas as pd
import numpy as np
from transformer_module import *
from keras.callbacks import EarlyStopping
import os
import unicodedata
import nltk.translate.bleu_score as bleu


os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

def main(args):

    model_path = args.model
    tokenizer_path = args.tokenizer
    dataset_path = args.dataset
    results_path = args.results_path

    MAX_LENGTH = int(args.max_length)
    BUFFER_SIZE = int(args.buffer_size)

    df = pd.read_csv(dataset_path)
    tokenizer = Tokenizer.load(tokenizer_path)


    test_indices = np.where(df['train_val_test'] == 'test')[0]
    test_dataset = df.loc[test_indices].reset_index(drop=True)

    tf.keras.backend.clear_session()

    VOCAB_SIZE = tokenizer.vocab_size 
    NUM_LAYERS = 2
    D_MODEL = 256
    NUM_HEADS = 8
    DFF = 512
    DROPOUT = 0.1

    model = Transformer_model(
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )
    model.summary()

    model.load_weights(model_path)

    bleu_4 = []
    for i in tqdm(range(len(test_dataset)), desc='Evaluate'):
        human_event, robot_response = test_dataset.loc[i, 'human_event'], test_dataset.loc[i, 'robot_response']
        predict = tokenizer.tokenize(tokenizer.decode(model.predict(human_event).numpy()))[1:]
        label = tokenizer.tokenize(robot_response)
        predict = tokenizer.convert_tokens_to_string(predict)
        predict = unicodedata.normalize('NFC', predict)
        label = tokenizer.convert_tokens_to_string(label)
        label = unicodedata.normalize('NFC', label)

        bleus = bleu.sentence_bleu(
            [predict],
            label,
            weights=(0,0,0,1)
        )

        bleu_4.append({
            'video_id' : test_dataset.loc[i, 'video_id'],
            'predict': predict,
            "label" : label,
            'bleu_4' : bleus,
        })


    df_blue_score = pd.DataFrame(bleu_4)
    print(f'Test Bleu-4 Score {df_blue_score["bleu_4"].mean()}')

    df_blue_score.to_csv(results_path, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    import argparse

    os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--results_path")
    parser.add_argument("--max_length", default=150)
    parser.add_argument("--buffer_size", default=200000)
    args = parser.parse_args()

    main(args)

    
