import time
import os
import argparse

import numpy as np

import torch 
from torch import nn
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torch.utils.data import DataLoader
from dataset_diffusionDB import DiffusionDB
from dataset_flickr30k import Flickr30k
from transformers import CLIPTextModel, CLIPTokenizer

from evaluate import clip_cos_similarity
global clip_tokenizer, clip_text_encoder, st_model

import sys
sys.path.append("/data/changl25/img2textModel/sentence-transformers/")
from sentence_transformers import SentenceTransformer

from utils import *

keyword_dics = {
"real" : "realistic, ",
"unreal" : "unrealistic, ",
"fantasy" : "fantasy, ",
"detailed" : "very detailed, ",
"res" : "high quality, ",
"focus" : "sharp focus, ",
"cinematic" : "cinematic lightening, ",
"painting" : "painting, ",
"digital" : "digital, ",
}

def sd_encoder(text, tokenizer, encoder, device):
    input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    embeddings = encoder(input_ids)[0]
    return embeddings

def sentence_cos_similarity(st_model, output_prompt, prompts):
    print(len(prompts), prompts[1])
    print(prompts)
    prompt_embedding = st_model.encode(prompts).flatten()
    output_embedding = st_model.encode(output_prompt).flatten()
    similarity = np.dot(prompt_embedding, output_embedding) / (np.linalg.norm(prompt_embedding) * np.linalg.norm(output_embedding))
    return similarity

def comb_output(images, model, classifiers):
    added_output = ""
    pixel_values = images.pop("pixel_values")
    with torch.no_grad():
        img_features = model.vision_model(pixel_values)
        image_input = img_features[0].view(1, -1)
        for key_word, classifier in classifiers.items():
            label = classifier(image_input)
            print(label)
            _, predicted = torch.max(label.data, 1)
            if predicted == 1:
                print("add label")
                added_output += keyword_dics[key_word]
    return added_output
    
def evaluate(peft_model, preprocessor, classifiers, data_loader, precision, \
    device, saved=False, saved_dir=None, max_new_tokens=50, repetition_penalty=1):
    loss = 0
    evaluate_text = []
    caption_text = []
    prompt_text = []
    peft_model.eval()
    
    loss_time = 0
    generate_time = 0
    load_time = 0
    
        
    if precision == "float32":
        st_model = SentenceTransformer('/data/changl25/img2textModel/all-MiniLM-L6-v2').to(device)
    
    if precision == "float16":
        st_model = SentenceTransformer('/data/changl25/img2textModel/all-MiniLM-L6-v2').to(device).half()
    
    peft_model.eval()
    st_model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            start_time = time.time()
            inputs, inputs_generator, prompt = data
            end_time3 = time.time()
            for key, value in inputs.items():
                if precision == "float16" and key == "pixel_values":
                    inputs[key] = value.to(device, torch.float16)
                    inputs_generator[key] = inputs_generator[key].to(device, torch.float16)
                else:
                    inputs[key] = value.to(device)
                    inputs_generator[key] = inputs_generator[key].to(device)
            out = peft_model.generate(**inputs_generator,
                                      max_new_tokens=max_new_tokens, 
                                      repetition_penalty=repetition_penalty,
                                      )
            out_text = preprocessor.decode(out[0], skip_special_tokens=True)
            added_lables = comb_output(inputs, peft_model, classifiers)
            print(added_lables)
            out_text = added_lables + out_text
            end_time2 = time.time()
            evaluate_text.append((out_text, prompt))
            generate_time += (end_time2  - end_time3) / len(prompt)
        start_time = time.time() 
        if saved == True:
            file = open(saved_dir, 'w')       
        for i in range(len(evaluate_text)):
            output_text, prompt = evaluate_text[i]
            caption_text.append(output_text)
            prompt_text.append(prompt)
            if saved == True:
                file.write(f"{output_text}_____{prompt}\n")
        sentence_sim = sentence_cos_similarity(st_model, caption_text, prompt_text)
        end_time = time.time()
        print(f"generate time: {generate_time / len(data_loader):.4f}, similar time: {(end_time - start_time) / len(caption_text):.4f}")
    return sentence_sim

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--model_load_dir", type=str, default="/data/changl25/img2textModel/blip_model")
    parser.add_argument("--model", type=str, default="blip2")
    parser.add_argument("--classifier_load_dir", type=str, default="/data/changl25/img2textModel/blip_model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--data_root_dir", type=str, default="/data/changl25/Diffusion2DB")
    parser.add_argument("--test_id", nargs='+', type=int, default=[1], help='test dataset id')
    parser.add_argument("--keyword", nargs='+', type=str, default=[], help='keyword')
    
    parser.add_argument("--save_dir", type=str, default="/home/changl25/Image-to-text-of-Stable-Diffusion/prompt_generate.csv")
    parser.add_argument("--precision", type=str, default="float32")
    args = parser.parse_args()
    return args

def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    precision = args.precision
    
    data_root_dir = args.data_root_dir
    test_root_dir = [os.path.join(data_root_dir, f"part-{part_id:06}") for part_id in args.test_id]
    
    save_dir = args.save_dir
    guide_text = "A photo of"
        
    if args.model == "blip":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        if args.pretrained:
            model = load_lora_blip2_model(BlipForConditionalGeneration, args.model_load_dir).to(device)
        else:
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    elif args.model == "blip2":
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        if args.pretrained:
            model = load_lora_blip2_model(Blip2ForConditionalGeneration, args.model_load_dir).to(device)
        else:
            model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco").to(device)

    if precision == "float16":
        model = model.half()
        
    classifiers = {}
    for key_word in args.keyword:
        classifier_path = os.path.join(args.classifier_load_dir, f"{key_word}_classifier.tar")
        checkpoint = torch.load(classifier_path)
        classifier = nn.Linear(677 * 1408, 2).to(device)
        if precision == "float16":
            classifier = classifier.half()
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifiers[key_word] = classifier
    test_dataset = DiffusionDB(test_root_dir, text = guide_text, transform=processor, max_length=args.max_length, regenerate=False)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)
    sen_sim = evaluate(model, processor, classifiers, test_loader, precision, device, \
        saved=True, saved_dir=save_dir, max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty)
    print(f"similarity: {sen_sim:.4f};")

if __name__ == "__main__":
    args = build_args()
    main(args)