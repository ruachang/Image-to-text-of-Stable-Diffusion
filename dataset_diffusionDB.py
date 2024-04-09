import os
import random

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
        root_img_dir_list,
        text=None,
        transform=None, 
        test=False,
    ):
        self.filename = []
        self.prompt = []
        self.text = text
        for root_img_dir in root_img_dir_list:
            json_file = os.path.basename(root_img_dir) + ".json"
            json_data = load(open(os.path.join(root_img_dir, json_file), 'r', encoding='utf8'))
        # self.root = root_img_dir
            keys = list(json_data.keys())
            random.seed(1024)
            selected_keys = random.sample(keys, len(json_data)//10)
            if test:
                img_data = {key: json_data[key] for key in selected_keys}
            else:
                train_keys = [key for key in keys if key not in selected_keys]
                img_data = {key: json_data[key] for key in train_keys}

            for img_name, img_info in img_data.items():
                self.filename.append(os.path.join(root_img_dir, img_name))
                self.prompt.append(img_info['p'])
        self.processor = transform
    def __getitem__(self, idx):
        img_path = self.filename[idx]
        img = Image.open(img_path)
        prompt = self.prompt[idx]
        if self.processor is not None and self.text is not None:
            inputs = self.processor(img, self.text, return_tensors="pt", padding="max_length")
            inputs_generate = self.processor(img, self.text, return_tensors="pt")
        elif self.processor is not None:
            inputs = self.processor(img, return_tensors="pt", padding="max_length")
            inputs_generate = self.processor(img, return_tensors="pt")
        for key, value in inputs.items():
            inputs[key] = value.squeeze(0)
        for key, value in inputs_generate.items():
            inputs_generate[key] = value.squeeze(0)
        return (inputs, inputs_generate, prompt)
    def __len__(self):
        return len(self.filename)
    def is_text_supervised(self):
        if self.text == None:
            return   False 
        else:
            return True
    
if __name__ == '__main__':
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    test_dataset = DiffusionDB(['/data/changl25/Diffusion2DB/part-000001', '/data/changl25/Diffusion2DB/part-000002'], text = "a photo of", transform=processor, test=True)
    test_loader = DataLoader.DataLoader(test_dataset,batch_size=8,shuffle=True)
    nxt = next(iter(test_loader))[0]["attention_mask"].shape
    print(nxt)