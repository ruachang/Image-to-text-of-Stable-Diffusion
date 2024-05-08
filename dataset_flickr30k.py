import os
import random

import numpy as np

import csv
import torch
import torch.utils.data as DataLoader
from json import load
from PIL import Image
from transformers import BlipProcessor, Blip2Processor

class Flickr30k(DataLoader.Dataset):
    def __init__(
        self,
        root_dir,
        text=None,
        transform=None, 
        split="train",
        data_split=1,
        max_length=None
    ):
        self.filename = []
        self.prompt = []
        self.text = text
        self.max_length = max_length
        csv_path = os.path.join(root_dir, "re-flickr30k.csv")
        img_root_path = os.path.join(root_dir, "flickr30k-images")
        
        keys = []
        values = []
        with open(csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                split_cls = row[2]
                if split_cls != split:
                    continue 
                caption_list = row[0].split(",")
                # caption_list = caption_list.split(",")
                img_name = row[3]
                img_path = os.path.join(img_root_path, img_name)
                caption_ele_lst = max(caption_list, key=len).split('"')
                caption =''.join(caption_ele_lst[i] for i in range(1, len(caption_ele_lst)))
                keys.append(img_path)
                values.append(caption)
            random.seed(1024)
            selected_keys = random.sample(list(range(len(keys))), len(keys)//data_split)

            self.filename = [keys[i] for i in selected_keys]
            self.prompt = [values[i] for i in selected_keys]
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
    
if __name__ == '__main__':
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
    test_dataset = Flickr30k('/data/changl25/flickr30k/', text = "a photo of", transform=processor, split="test")
    test_loader = DataLoader.DataLoader(test_dataset,batch_size=8,shuffle=True)
    nxt = next(iter(test_loader))[0]["attention_mask"].shape
    print(nxt)