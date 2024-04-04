import os

import cv2
import numpy as np

import torch.utils.data as DataLoader
import torchvision.transforms as transforms
from json import load
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class DiffusionDB(DataLoader.Dataset):
    def __init__(
        self,
        root_img_dir,
        text=None,
        transform=None, 
    ):
        json_file = os.path.basename(root_img_dir) + ".json"
        json_data = load(open(os.path.join(root_img_dir, json_file), 'r', encoding='utf8'))
        self.root = root_img_dir
        self.filename = []
        self.prompt = []
        self.text = text
        for img_name, img_info in json_data.items():
            self.filename.append(img_name)
            self.prompt.append(img_info['p'])
        self.processor = transform
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.filename[idx])
        img = Image.open(img_path)
        prompt = self.prompt[idx]
        if self.processor is not None and self.text is not None:
            inputs = self.processor(img, self.text, return_tensors="pt")
        elif self.processor is not None:
            inputs = self.processor(img, return_tensors="pt")
        for key, value in inputs.items():
            inputs[key] = value.squeeze(0)
        return (inputs, prompt)
    def __len__(self):
        return len(self.filename)
    
if __name__ == '__main__':
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    test_dataset = DiffusionDB('/data/changl25/DiffusionDB/part-000001', text = "a photo of", transform=processor)
    test_loader = DataLoader.DataLoader(test_dataset,batch_size=8,shuffle=True)
    nxt = next(iter(test_loader))[0]["attention_mask"].shape
    print(nxt)