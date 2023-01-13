import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

IMG_SIZE=224
#IMG_SIZE=256
#IMG_SIZE=320
#IMG_SIZE=260

class EmotionDataset(Dataset):
    def __init__(self, path=None, data=None, transforms=None):
        if path is None:
            self.data = data
        else:
            self.data = pd.read_csv(path)
        self.targets = list(map(lambda x : x, self.data['class'].to_list()))
        self.classes = np.unique(self.targets)
        self.transforms = transforms
    
        self.class_to_index={
            1:0,
            2:1,
            3:2,
            4:3,
            5:4,
        }

        self.classname_to_index={
            '기쁨':0,
            '화남':1,
            '놀람':2,
            '무표정':3,
            '모름':4,
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'image_path']
        video_id = self.data.loc[idx, 'video_id']
        #label = int(self.data.loc[idx, 'class'])
        #label = self.class_to_index[label]
        label = self.data.loc[idx, 'class_name']
        label = self.classname_to_index[label]
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return idx, img, label, video_id, img_path


emotion_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)
