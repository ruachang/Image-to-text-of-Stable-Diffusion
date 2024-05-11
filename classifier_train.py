import os
import argparse

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
import numpy as np

import wandb

from dataset_diffusionDB import DiffusionDB, DiffusionDB_label
from dataset_flickr30k import Flickr30k
from evaluate import evaluate, sentence_cos_similarity
from utils import *
from scheduler import WarmupCosineLR
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--model", type=str, default="blip2")
    parser.add_argument("--model_load_dir", type=str, default="/data/changl25/img2textModel/blip_model")
    parser.add_argument("--classifier_load_dir", type=str, default="/data/changl25/img2textModel/classifier")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    
    parser.add_argument("--data_root_dir", type=str, default="/data/changl25/Diffusion2DB")
    parser.add_argument("--label", type=str)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--train_id", nargs='+', type=int, default=[2, 3], help='trained dataset ids')
    parser.add_argument("--test_id", nargs='+', type=int, default=[1], help='test dataset ids')
    
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test_label", nargs='+', type=str, default=[])
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

def evaluate_classifier(model, classifier, data_loader, device):
    classifier.eval()
    loss = 0
    accurate = 0
    total = 0
    mse_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            images, labels = data
            for key, value in images.items():
                images[key] = value.to(device)
            pixel_values = images.pop("pixel_values")
            with torch.no_grad():
                img_features = model.vision_model(pixel_values)
                img_feature_input = img_features[0].view(data_loader.batch_size, -1)
            outputs = classifier(img_feature_input)
            labels = labels.to(device)
            loss += mse_loss(outputs, labels) / len(labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accurate += (predicted == labels).sum().item()
    return loss / len(data_loader), accurate / total
def train_classifier(model, classifier, train_loader, validate_loader, test_loader, epochs, optimizer, scheduler, \
    precision, model_save_dir, key_word, use_wandb):
    best_test_acc = 0
    mse_loss = nn.CrossEntropyLoss()
    if precision == "float16":
        scaler = GradScaler(enabled=True)
    for epoch in range(epochs):
        classifier.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            for key, value in images.items():
                images[key] = value.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if precision == "float16":
                 with autocast(enabled=True):
                    pixel_values = images.pop("pixel_values")
                    with torch.no_grad():
                        img_features = model.vision_model(pixel_values)
                        img_feature_input = img_features[0].view(train_loader.batch_size, -1)
                    outputs = classifier(img_feature_input)
                    loss = mse_loss(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                pixel_values = images.pop("pixel_values")
                with torch.no_grad():
                    img_features = model.vision_model(pixel_values)
                outputs = classifier(img_features)
                loss = mse_loss(outputs, labels)
                loss.backward()
                optimizer.step()
            if scheduler != None:    
                scheduler.step()
            if (i + 1) % (len(train_loader) // 3) == 0:
                print(f"Epoch {epoch}: {i} / {len(train_loader)} {(loss / len(labels)):.4f}")
        validate_loss, validate_acc = evaluate_classifier(model, classifier, validate_loader, device)
        test_loss, test_acc = evaluate_classifier(model, classifier, test_loader, device)
        print(f"Epoch {epoch}: validate: loss {validate_loss:.4f}; acc {validate_acc}")
        print(f"Epoch {epoch}: test:     loss {test_loss:.4f}; acc {test_acc}")
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            save_classifier(model_save_dir, classifier, "best", key_word)
        if use_wandb:
            wandb.log({"validate_loss": validate_loss}, step = epoch + 1)
            wandb.log({"test_loss": test_loss}, step = epoch + 1)
            wandb.log({"validate_acc": validate_acc}, step = epoch + 1)
            wandb.log({"test_acc": test_acc}, step = epoch + 1)
        # if epoch in [0, 5, 10, 20, 30, 40]:
        save_classifier(model_save_dir, classifier, str(epoch), key_word)
    print("Trained down!")    
    
def main(args):
    if args.test:
        batch_size = args.batch_size
        precision = args.precision
        
        data_root_dir = args.data_root_dir
        test_root_dir = [os.path.join(data_root_dir, f"part-{part_id:06}") for part_id in args.test_id]
        
        guide_text = "A photo of"
        # guide_text = None
        
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
        model = model.to(device).eval()
        file = open(args.save_dir, 'w')
        file.write(f"Classifier path {args.save_dir}")
        for key_word in args.test_label:
            classifier = nn.Linear(677 * 1408, 2).to(device)
            classifier_path = os.path.join(args.classifier_load_dir, f"{key_word}_classifier.tar")
            checkpoint = torch.load(classifier_path)

            classifier.load_state_dict(checkpoint['model_state_dict'])
            test_dataset = DiffusionDB_label(key_word, test_root_dir, text = guide_text, transform=processor, max_length=args.max_length)
            test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

            loss, acc = evaluate_classifier(model, classifier, test_loader, device)
            print("Evaluate after training")
            print(f"Final loss: {loss:.4f}, acc: {acc:.4f}")
            file.write(f"{key_word}: {acc} \n")
        file.close()
    else:
        batch_size = args.batch_size
        epochs = args.epochs
        precision = args.precision
        label = args.label
        
        data_root_dir = args.data_root_dir
        train_root_dir = [os.path.join(data_root_dir, f"part-{part_id:06}") for part_id in args.train_id]
        test_root_dir = [os.path.join(data_root_dir, f"part-{part_id:06}") for part_id in args.test_id]
        
        model_save_dir = os.path.join(args.model_save_dir)
        os.makedirs(model_save_dir, exist_ok=True)
        guide_text = "A photo of"
        # guide_text = None
        
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
        model = model.to(device).eval()
        classifier = nn.Linear(677 * 1408, 2).to(device)
        test_dataset = DiffusionDB_label(label, test_root_dir, text = guide_text, transform=processor, max_length=args.max_length)
        train_dataset = DiffusionDB_label(label, train_root_dir, text = guide_text, transform=processor, max_length=args.max_length)
        validate_dataset = DiffusionDB_label(label, train_root_dir, text = guide_text, transform=processor, max_length=args.max_length, test=True)
        
        validate_loader = DataLoader(validate_dataset,batch_size=batch_size,shuffle=False)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        if args.warmup:
            scheduler = WarmupCosineLR(optimizer, warmup_steps=args.warmup_steps, total_steps=len(train_loader)*epochs, \
                warmup_init_lr=args.warmup_init_lr, max_lr=args.init_lr, min_lr=args.warmup_min_lr)
        else:
            scheduler = None 
        if args.wandb: 
            config = {
                "epochs": epochs,
                "batch size": batch_size,
                "train id": args.train_id,
                "saved_model": model_save_dir,
                "text": guide_text
            }
            init_wandb(config, classifier, f"classifier{epochs}_{batch_size}")
        train_classifier(model, classifier, train_loader, validate_loader, test_loader, epochs, optimizer, scheduler, precision, \
            model_save_dir=model_save_dir, key_word=label, use_wandb=args.wandb)
        loss, acc = evaluate_classifier(model, classifier, test_loader, device)
        print("Evaluate after training")
        print(f"Final loss: {loss:.4f}, acc: {acc:.4f}")
        save_classifier(model_save_dir, classifier, str(epochs), label)
        if args.wandb:
            wandb.finish()
if __name__ == "__main__":
    args = build_args()
    main(args)
