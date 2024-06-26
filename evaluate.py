import time
import os
import argparse

import numpy as np

import torch 

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from torch.utils.data import DataLoader
from dataset_diffusionDB import DiffusionDB
from dataset_flickr30k import Flickr30k
from transformers import CLIPTextModel, CLIPTokenizer
global clip_tokenizer, clip_text_encoder, st_model

import sys
sys.path.append("/data/changl25/img2textModel/sentence-transformers/")
from sentence_transformers import SentenceTransformer

from utils import *

def sd_encoder(text, tokenizer, encoder, device):
    input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    embeddings = encoder(input_ids)[0]
    return embeddings

def clip_cos_similarity(clip_tokenizer, clip_text_encoder, output_prompt, prompts, device):
    caption_embeddings = sd_encoder(output_prompt, clip_tokenizer, clip_text_encoder, device)
    prompt_embeddings = sd_encoder(prompts, clip_tokenizer, clip_text_encoder, device)
    prompt_embeds_flat = prompt_embeddings.view(prompt_embeddings.size(0), -1)
    caption_embeds_flat = caption_embeddings.view(caption_embeddings.size(0), -1)
    prompt_embeds = prompt_embeds_flat / prompt_embeds_flat.norm(dim=1, keepdim=True)
    caption_embeds = caption_embeds_flat / caption_embeds_flat.norm(dim=1, keepdim=True)
    cos_similarity = (torch.matmul(prompt_embeds, caption_embeds.t())).mean()
    return cos_similarity

def sentence_cos_similarity(st_model, output_prompt, prompts):
    prompt_embedding = st_model.encode(prompts).flatten()
    output_embedding = st_model.encode(output_prompt).flatten()
    similarity = np.dot(prompt_embedding, output_embedding) / (np.linalg.norm(prompt_embedding) * np.linalg.norm(output_embedding))
    return similarity

def evaluate(peft_model, preprocessor, data_loader, text_flag, precision, device, saved=False, saved_dir=None, max_length=512, max_new_tokens=50, repetition_penalty=1):
    loss = 0
    evaluate_text = []
    caption_text = []
    prompt_text = []
    peft_model.eval()
    
    loss_time = 0
    generate_time = 0
    load_time = 0
    
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    clip_text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
    if precision == "float32":
        st_model = SentenceTransformer('/data/changl25/img2textModel/all-MiniLM-L6-v2').to(device)
    
    if precision == "float16":
        st_model = SentenceTransformer('/data/changl25/img2textModel/all-MiniLM-L6-v2').to(device).half()
    
    peft_model.eval()
    clip_text_encoder.eval()
    st_model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            start_time = time.time()
            inputs, inputs_generator, prompt = data
            for key, value in inputs.items():
                if precision == "float16" and key == "pixel_values":
                    inputs[key] = value.to(device, torch.float16)
                    inputs_generator[key] = inputs_generator[key].to(device, torch.float16)
            
                else:
                    inputs[key] = value.to(device)
                    inputs_generator[key] = inputs_generator[key].to(device)
                labels = torch.tensor(preprocessor.tokenizer(text=prompt, padding="max_length", truncation=True, max_length=max_length)["input_ids"]).to(device)
            end_time3 = time.time()
            if text_flag:
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                attention_mask = inputs["attention_mask"]
                outputs = peft_model(input_ids=input_ids, pixel_values=pixel_values,attention_mask=attention_mask, labels=labels)
            else:
                pixel_values = inputs["pixel_values"]
                outputs = peft_model(pixel_values=pixel_values, labels=labels)
            if torch.isnan(outputs.loss ).any():
                print(prompt)
                continue
            # raise ValueError(f'NaN detected in loss of {labels}')
            loss += outputs.loss / len(prompt)
            end_time1 = time.time()
            out = peft_model.generate(**inputs_generator,
                                      max_new_tokens=max_new_tokens, 
                                      repetition_penalty=repetition_penalty,
                                      )
            out_text = preprocessor.batch_decode(out, skip_special_tokens=True)
            end_time2 = time.time()
            evaluate_text.append((out_text, prompt))
            loss_time += (end_time1  - end_time3) / len(prompt)
            generate_time += (end_time2  - end_time1) / len(prompt)
            load_time += (end_time3 - start_time) / len(prompt)
        start_time = time.time() 
        if saved == True:
            file = open(saved_dir, 'w')       
        for i in range(len(evaluate_text)):
            output_text_batch, prompt_batch = evaluate_text[i]
            for j in range(len(output_text_batch)):
                output_text, prompt = output_text_batch[j], prompt_batch[j]
                caption_text.append(output_text)
                prompt_text.append(prompt)
                if saved == True:
                    file.write(f"{output_text}_____{prompt}\n")
        clip_sim = clip_cos_similarity(clip_tokenizer, clip_text_encoder, caption_text, prompt_text, device)
        sentence_sim = sentence_cos_similarity(st_model, caption_text, prompt_text)
        end_time = time.time()
        print(f"load time: {load_time / len(data_loader):.4f}, loss time: {loss_time / len(data_loader):.4f}, generate time: {generate_time / len(data_loader):.4f}, similar time: {(end_time - start_time) / len(caption_text):.4f}")
        
    return loss / len(data_loader), clip_sim, sentence_sim

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--model_load_dir", type=str, default="/data/changl25/img2textModel/blip_model")
    parser.add_argument("--model", type=str, default="blip2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="Diffusion2DB")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)
    parser.add_argument("--data_root_dir", type=str, default="/data/changl25/Diffusion2DB")
    parser.add_argument("--test_id", nargs='+', type=int, default=[1], help='test dataset id')
    
    parser.add_argument("--save_dir", type=str, default="/home/changl25/Image-to-text-of-Stable-Diffusion/prompt_generate.csv")
    parser.add_argument("--precision", type=str, default="float32")
    args = parser.parse_args()
    return args

def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = args.batch_size
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
    if args.dataset == "Diffusion2DB":
        test_dataset = DiffusionDB(test_root_dir, text = guide_text, transform=processor, max_length=args.max_length, regenerate=False)
    elif args.dataset == "flickr30k":
        test_dataset = Flickr30k(data_root_dir, text = guide_text, transform=processor, split="test", max_length=args.max_length)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    loss, clip_sim, sen_sim = evaluate(model, processor, test_loader, test_dataset.is_text_supervised(), precision, device, \
        saved=True, saved_dir=save_dir, max_length=args.max_length, max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty)
    print(f"Evaluate losss: {loss:.4f}; similarity: {sen_sim:.4f}; clip similarity: {clip_sim:.4f}")

if __name__ == "__main__":
    args = build_args()
    main(args)