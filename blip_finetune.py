import os
import argparse

import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import numpy as np

import wandb

from dataset_diffusionDB import DiffusionDB
from dataset_flickr30k import Flickr30k
from evaluate import evaluate
from utils import *
from scheduler import WarmupCosineLR
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--model", type=str, default="blip2")
    parser.add_argument("--model_load_dir", type=str, default="/data/changl25/img2textModel/blip_model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    
    parser.add_argument("--data_root_dir", type=str, default="/data/changl25/Diffusion2DB")
    parser.add_argument("--dataset", type=str, default="Diffusion2DB")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--train_id", nargs='+', type=int, default=[2, 3], help='trained dataset ids')
    parser.add_argument("--test_id", nargs='+', type=int, default=[1], help='test dataset ids')
    
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_modules", nargs='+', type=str, default=[])
    
    parser.add_argument("--warmup", action='store_true')
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--warmup_init_lr", type=float, default=1e-8)
    parser.add_argument("--init_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_min_lr", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    
    parser.add_argument("--model_save_dir", type=str, default="/data/changl25/img2textModel/blip_model")
    parser.add_argument("--save_dir", type=str, default="/home/changl25/Image-to-text-of-Stable-Diffusion/prompt_generate.csv")
    parser.add_argument("--precision", type=str, default="float32")
    parser.add_argument("--wandb", action='store_true')
    args = parser.parse_args()
    return args

def train_manual(peft_model, preprocessor, train_loader, validate_loader, test_loader, epochs, optimizer, scheduler, \
    precision, text_flag, save_dir, use_wandb, max_length, max_new_tokens, num_beams):
    best_validate_loss = float("inf")
    if precision == "float16":
        scaler = GradScaler(enabled=True)
    for epoch in range(epochs):
        peft_model.train()
        for i, data in enumerate(train_loader):
            inputs, _, prompt = data
            optimizer.zero_grad()
            for key, value in inputs.items():
                    inputs[key] = value.to(device)
                    labels = torch.tensor(preprocessor.tokenizer(text=prompt, padding="max_length", truncation=True, max_length=max_length)["input_ids"]).to(device)
            optimizer.zero_grad()
            if precision == "float16":
                 with autocast(enabled=True):
                    if text_flag:
                        input_ids = inputs.pop("input_ids")
                        pixel_values = inputs.pop("pixel_values")
                        attention_mask = inputs.pop("attention_mask")
                        outputs = peft_model(input_ids=input_ids, pixel_values=pixel_values,attention_mask=attention_mask,labels=labels)
                    else:
                        pixel_values = inputs.pop("pixel_values")
                        outputs = peft_model(pixel_values=pixel_values, labels=labels)
            
                    loss = outputs.loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if text_flag:
                    input_ids = inputs.pop("input_ids")
                    pixel_values = inputs.pop("pixel_values")
                    attention_mask = inputs.pop("attention_mask")
                    outputs = peft_model(input_ids=input_ids, pixel_values=pixel_values,attention_mask=attention_mask,labels=labels)
                else:
                    pixel_values = inputs.pop("pixel_values")
                    outputs = peft_model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            if scheduler != None:    
                scheduler.step()
            if (i + 1) % (len(train_loader) // 3) == 0:
                print(f"Epoch {epoch}: {i} / {len(train_loader)} {(loss / len(prompt)):.4f}")
        validate_loss, validate_clip_sim, validate_similarity = evaluate(peft_model, preprocessor,validate_loader, text_flag, precision, device)
        test_loss, test_clip_sim, test_similarity = evaluate(peft_model, preprocessor,test_loader, text_flag, precision, device)
        print(f"Epoch {epoch}: validate: loss {validate_loss:.4f}; similarity: {validate_similarity:.4f}; clip similarity: {validate_clip_sim:.4f}")
        print(f"Epoch {epoch}: test:     loss {test_loss:.4f}; similarity: {test_similarity:.4f}; clip similarity: {test_clip_sim:.4f}")
        if validate_loss < best_validate_loss:
            best_validate_loss = validate_loss
            save_model(save_dir, peft_model, "best")
        if use_wandb:
            wandb.log({"validate_loss": validate_loss}, step = epoch + 1)
            wandb.log({"validate_clip_sim": validate_clip_sim}, step = epoch + 1)
            wandb.log({"validate_similarity": validate_similarity}, step = epoch + 1)
            wandb.log({"test_loss": test_loss}, step = epoch + 1)
            wandb.log({"test_clip_sim": test_clip_sim}, step = epoch + 1)
            wandb.log({"test_similarity": test_similarity}, step = epoch + 1)
        if epoch in [0, 5, 10, 20, 30, 40]:
            save_model(save_dir, peft_model, str(epoch))
    print("Trained down!")     

def main(args):
    batch_size = args.batch_size
    epochs = args.epochs
    precision = args.precision
    
    data_root_dir = args.data_root_dir
    train_root_dir = [os.path.join(data_root_dir, f"part-{part_id:06}") for part_id in args.train_id]
    test_root_dir = [os.path.join(data_root_dir, f"part-{part_id:06}") for part_id in args.test_id]
    
    save_dir = args.save_dir
    model_save_dir = args.model_save_dir
    os.makedirs(model_save_dir, exist_ok=True)
    guide_text = "A photo of"
    
    if args.model == "blip":
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        if args.pretrained:
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            model = load_lora_blip2_model(BlipForConditionalGeneration, args.model_load_dir).to(device)
        else:
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    elif args.model == "blip2":
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        if args.pretrained:
            model = load_lora_blip2_model(Blip2ForConditionalGeneration, args.model_load_dir).to(device)
        else:
            model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b-coco").to(device)

    lora_configs = {}
    lora_configs["lora_rank"] = args.lora_rank
    lora_configs["lora_alpha"] = args.lora_alpha
    lora_configs["lora_bias"] = args.lora_bias
    lora_configs["lora_dropout"] = args.lora_dropout
    lora_configs["target_modules"] = args.lora_modules
    
    peft_model = get_lora_model(model, lora_configs)
    peft_model = peft_model.to(device)
    
    if args.dataset == "Diffusion2DB":
        test_dataset = DiffusionDB(test_root_dir, text = guide_text, transform=processor, max_length=args.max_length, regenerate=False)
        train_dataset = DiffusionDB(train_root_dir, text = guide_text, transform=processor, max_length=args.max_length)
        validate_dataset = DiffusionDB(train_root_dir, text = guide_text, transform=processor, max_length=args.max_length, test=True)
    elif args.dataset == "flickr30k":
        test_dataset = Flickr30k(data_root_dir, text = guide_text, transform=processor, split="test", max_length=args.max_length)
        train_dataset = Flickr30k(data_root_dir, text = guide_text, transform=processor, split="train", max_length=args.max_length,  data_split=args.split)
        validate_dataset = Flickr30k(data_root_dir, text = guide_text, transform=processor, split="val", max_length=args.max_length, data_split=args.split)
    
    validate_loader = DataLoader(validate_dataset,batch_size=batch_size,shuffle=False)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    if args.warmup:
        scheduler = WarmupCosineLR(optimizer, warmup_steps=args.warmup_steps, total_steps=len(train_loader)*epochs, \
            warmup_init_lr=args.warmup_init_lr, max_lr=args.init_lr, min_lr=args.warmup_min_lr)
    else:
        scheduler = None 
    if args.wandb: 
        config = {
            "epochs": epochs,
            "batch size": batch_size,
            "lora rank": args.lora_rank,
            "lora alpha": args.lora_alpha,
            "train id": args.train_id,
            "saved_model": model_save_dir,
            "text": guide_text
        }
        init_wandb(config, peft_model, f"blip{epochs}_{batch_size}")
    train_manual(peft_model, processor, train_loader, validate_loader, test_loader, epochs, optimizer, scheduler, precision, train_dataset.is_text_supervised(), model_save_dir, args.wandb, \
        max_length=args.max_length, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
    loss, clip_sim, sen_sim = evaluate(peft_model, processor, test_loader, test_dataset.is_text_supervised(), precision, device, saved=True, saved_dir=save_dir, \
        max_length=args.max_length, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
    print("Evaluate after training")
    print(f"Final loss: {loss:.4f}; similarity: {sen_sim:.4f}; clip similarity: {clip_sim:.4f}")
    save_model(model_save_dir, peft_model, str(epochs))
    if args.wandb:
        wandb.finish()
if __name__ == "__main__":
    args = build_args()
    main(args)
