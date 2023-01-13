from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from transformers import AutoTokenizer
import json
from glob import glob
import os

class Tokenizer():
    def load(path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.bos_token = '[BOS]'
        tokenizer.eos_token = '[EOS]'
        return tokenizer
    
    def train(path, data_root, vocab_size=3000, limit_alphabet=6000, min_frequency=1):
        model_path = path
        
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        user_defined_symbols = ["[BOS]","[EOS]"]

        special_tokens = special_tokens + user_defined_symbols

        paths = glob(os.path.join(data_root, "**/*.txt"), recursive=True)
        

        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True, 
            lowercase=True,
            wordpieces_prefix="##"
        )

        tokenizer.train(
            files=paths,
            limit_alphabet=limit_alphabet,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=special_tokens
        )

        vocab_path = os.path.join(model_path, "tok_added-ch-{}-wpm-{}".format(limit_alphabet, vocab_size))

        if not os.path.exists(model_path):
            os.mkdir(model_path)
        tokenizer.save(vocab_path, True)

        vocab_file = os.path.join(model_path, 'wordpiece_vocab.txt')

        f = open(vocab_file,'w',encoding='utf-8')
        with open(vocab_path, encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            for item in json_data["model"]["vocab"].keys():
                f.write(item+'\n')

            f.close()

        tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True) 
        tokenizer.add_special_tokens({'additional_special_tokens':user_defined_symbols})

        tokenizer.save_pretrained(model_path)


if __name__ == '__main__':
    import pandas as pd 
    from tqdm import tqdm
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
    )
    parser.add_argument(
        "--tokenizer",
    )
    parser.add_argument(
        "--temp_dir",
    )
    parser.add_argument(
        "--vocap_size",
        default=8000
    )
    parser.add_argument(
        "--limit_alphabet",
        default=16000
    )
    parser.add_argument(
        "--min_frequency",
        default=1
    )
    args = parser.parse_args()
    dataset_path = args.dataset
    tokenizer_path = args.tokenizer
    temp_dir = args.temp_dir
    vocap_size = int(args.vocap_size)
    limit_alphabet = int(args.limit_alphabet)
    min_frequency = int(args.min_frequency)


    datasets = pd.read_csv(dataset_path)
    for i in tqdm(range(len(datasets))):
        video_id = datasets.loc[i, 'video_id']
        human_event = datasets.loc[i, 'human_event']
        robot_response = datasets.loc[i, 'robot_response']
        robot_response = datasets.loc[i, 'robot_response'].replace('[SEP]', '')
        robot_response = datasets.loc[i, 'robot_response'].replace('[SEP]', '')
        with open(f'{temp_dir}/{video_id}.txt', 'w') as f:
            f.write(human_event+'\n')
            f.write(robot_response+'\n')
    
    Tokenizer.train(tokenizer_path, temp_dir, vocab_size=vocap_size, limit_alphabet=limit_alphabet, min_frequency=min_frequency)

    
