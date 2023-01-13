import tensorflow as tf
from glob import glob
import json
from datetime import datetime
from tqdm.notebook import tqdm
from tokenizer import Tokenizer
import pandas as pd
import numpy as np
from transformer_module import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os


def main(args):
    
    model_path = args.model
    tokenizer_path = args.tokenizer
    dataset_path = args.dataset

    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epoch)
    MAX_LENGTH = int(args.max_length)
    BUFFER_SIZE = int(args.buffer_size)

    df = pd.read_csv(dataset_path)
    tokenizer = Tokenizer.load(tokenizer_path)

    train_indices = np.where(df['train_val_test'] == 'train')[0]
    valid_indices = np.where(df['train_val_test'] == 'validation')[0]

    train_dataset = df.loc[train_indices].reset_index(drop=True)
    valid_dataset = df.loc[valid_indices].reset_index(drop=True)

    train_questions, train_answers = train_dataset['human_event'], train_dataset['robot_response']
    train_questions = list(map(lambda x : preprocess_sentence(x), train_questions))
    train_answers = list(map(lambda x : preprocess_sentence(x), train_answers))
    train_questions = tokenize_and_padding(train_questions, tokenizer, MAX_LENGTH)
    train_answers = tokenize_and_padding(train_answers, tokenizer, MAX_LENGTH)

    valid_questions, valid_answers = valid_dataset['human_event'], valid_dataset['robot_response']
    valid_questions = list(map(lambda x : preprocess_sentence(x), valid_questions))
    valid_answers = list(map(lambda x : preprocess_sentence(x), valid_answers))
    valid_questions = tokenize_and_padding(valid_questions, tokenizer, MAX_LENGTH)
    valid_answers = tokenize_and_padding(valid_answers, tokenizer, MAX_LENGTH)

    train_dataset = dataset_slices(train_questions, train_answers, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)
    valid_dataset = dataset_slices(valid_questions, valid_answers, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE)

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

    #model_checkpoint_callback = ModelCheckpoint(
    #    filepath=checkpoint_filepath,
    #    save_weights_only=True,
    #    monitor='val_loss',
    #    mode='auto',
    #    save_best_only=True
    #)


    early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, mode = 'auto')

    def loss_function(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_mean(loss)
    
    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    def accuracy(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    history = model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=EPOCHS, callbacks=[early_stopping])

    model.save_weights(model_path)


if __name__ == '__main__':
    import argparse 
    
    os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--epoch", default=60)
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--max_length", default=150)
    parser.add_argument("--buffer_size", default=200000)
    args = parser.parse_args()

    main(args)
