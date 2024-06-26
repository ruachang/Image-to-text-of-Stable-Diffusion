import os
import random

import cv2
import numpy as np

import torch
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
        regenerate=True,
        max_length=None
    ):
        self.filename = []
        self.prompt = []
        self.text = text
        self.max_length = max_length
        for root_img_dir in root_img_dir_list:
            if regenerate:
                json_file = "re-" + os.path.basename(root_img_dir) + ".json"
            else:
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
                if regenerate:
                    self.prompt.append(img_info)
                else:
                    self.prompt.append(img_info['p'])
        self.processor = transform
    def __getitem__(self, idx):
        img_path = self.filename[idx]
        img = Image.open(img_path)
        prompt = self.prompt[idx]
        if self.processor is not None and self.text is not None:
            if self.max_length == None:
                inputs = self.processor(img, self.text, return_tensors="pt", padding="max_length")
            else:
                inputs = self.processor(img, self.text, return_tensors="pt", padding="max_length", max_length=self.max_length)
            inputs_generate = self.processor(img, self.text, return_tensors="pt")
        elif self.processor is not None:
            if self.max_length == None:
                inputs = self.processor(img, return_tensors="pt", padding="max_length")
            else:
                inputs = self.processor(img, return_tensors="pt", padding="max_length", max_length=self.max_length)
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
        
class DiffusionDB_label(DataLoader.Dataset):
    def __init__(
        self,
        key_word,
        root_img_dir_list,
        text=None,
        transform=None, 
        test=False,
        max_length=None
    ):
        self.filename = []
        self.text = text
        self.key_word = key_word
        self.label = []
        self.max_length = max_length
        for root_img_dir in root_img_dir_list:
            json_file = f"{key_word}-" + os.path.basename(root_img_dir) + ".json"
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
                self.label.append(img_info)
        self.processor = transform
    def __getitem__(self, idx):
        img_path = self.filename[idx]
        img = Image.open(img_path)
        label = torch.tensor(self.label[idx][0])
        if self.processor is not None and self.text is not None:
            if self.max_length == None:
                inputs = self.processor(img, self.text, return_tensors="pt", padding="max_length")
            else:
                inputs = self.processor(img, self.text, return_tensors="pt", padding="max_length", max_length=self.max_length)
        elif self.processor is not None:
            if self.max_length == None:
                inputs = self.processor(img, return_tensors="pt", padding="max_length")
            else:
                inputs = self.processor(img, return_tensors="pt", padding="max_length", max_length=self.max_length)
        for key, value in inputs.items():
            inputs[key] = value.squeeze(0)
        # print(label)
        return (inputs, label)
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