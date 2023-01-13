import numpy as np
import pandas as pd
import torch
from emotion_dataset import EmotionDataset, emotion_transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def main(args):
    emotaion_dataset_path = args.dataset
    model_path = args.model
    results_path = args.results_path
    batch_size = int(args.batch_size)
    num_workers =int(args.num_workers)
    pin_memory = True
    device = 'cuda'

    model = torch.load(model_path)
    model = model.eval()
    
    df_face_dataset = pd.read_csv(emotaion_dataset_path)
    df_face_testset = df_face_dataset[df_face_dataset['train_val_test'] == 'test'].reset_index(drop=True)
    df_face_testset['image_path'] = df_face_testset['image_path'].map(lambda x : x.replace('/mnt/hdd18t/', '/mnt/ssd4t/'))

    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if 'cuda' in device else {}
    test_dataset = EmotionDataset(data=df_face_testset, transforms=emotion_transforms)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs) 

    results=[]
    for idx, data, label, video_id, image_path in tqdm(test_loader, desc='Test'):
        data = data.to(device)
        output = model(data)
        predict = output.argmax(dim=1).to('cpu')
        results.append(pd.DataFrame({
            "video_id":video_id, 'image_path':image_path,
            'predict':predict, 'label':label
        }))
    results = pd.concat(results)

    results.to_csv(results_path, index=False)

    acc = accuracy_score(results['predict'], results['label'])
    print(f"testset emotion accuracy score : {acc}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--results_path")
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_workers", default=1)
    args = parser.parse_args()
    
    main(args)
