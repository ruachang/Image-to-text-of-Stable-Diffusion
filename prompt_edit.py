from json import load
import json
import os
import re
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image
import numpy as np
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

prompt_tokenizer = AutoTokenizer.from_pretrained('alibaba-pai/pai-bloom-1b1-text2prompt-sd-v2')
prompt_model = AutoModelForCausalLM.from_pretrained('alibaba-pai/pai-bloom-1b1-text2prompt-sd-v2').to(device).eval()
clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device).eval()
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

TEMPLATE_V2 = 'Convert too complex image description into more brief prompt. \
Prompts are formatted as content description and multiple related tags separated by commas, \
you should add commas to seperate content description and tags\
\n\
### Input: {raw_prompt}\n### Output: '

def get_json_data(json_path):
    json_data = load(open(json_path, 'r', encoding='utf8'))
    json_keys = list(json_data.keys())
    json_values = list(json_data.values())
    return json_keys, json_values
def get_part_json(root_img_dir, id):
    work_file = os.path.join(root_img_dir, f"part-{id:06}")
    json_file = os.path.basename(work_file) + ".json"
    
    json_data = load(open(os.path.join(work_file, json_file), 'r', encoding='utf8'))
    # * name of all images
    json_keys = list(json_data.keys())
    img_data = {key: json_data[key] for key in list(json_data.keys())}
    # * value of all prompts
    prompt = []
    for img_name, img_info in img_data.items():
        prompt.append(img_info['p'])
    return json_keys, prompt, work_file

def image_txt_sim(img_root_path, imgs, text):
# * Use Clip to select the imgs which is most similiar to the text
# Assume 'image' is a PIL Image object and 'text' is a string
    with torch.no_grad():
        if len(imgs) > 1:
            img_lst = []
            for img in imgs:
                img_lst.append(Image.open(os.path.join(img_root_path, img)))
            inputs = clip_processor(text=text, images=img_lst, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            max_prob, max_index = torch.max(logits_per_image, dim=0) 
            return imgs[max_index]
        else:
            img = Image.open(os.path.join(img_root_path, imgs[0]))
            inputs = clip_processor(text=text, images=img, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            max_prob, max_index = torch.max(logits_per_image, dim=1) 
            return max_prob, text[max_index]
def filter_same_image(img_root_path, img_lst, prompt_lst):
    prompt_dic = {}
    filter_dic = {}
    for ids in range(len(prompt_lst)):
        prompt = prompt_lst[ids]
        img = img_lst[ids]
        if prompt not in prompt_dic.keys():
            prompt_dic[prompt] = []
        prompt_dic[prompt].append(img)
        
    for prompt in prompt_dic.keys():
        if len(prompt_dic[prompt]) > 1:
            bst_img = image_txt_sim(img_root_path, prompt_dic[prompt], prompt)
        else:
            bst_img = prompt_dic[prompt][0]
        filter_dic[bst_img] = prompt
    return filter_dic

def delete_short_prompts(prompt_dic, threshold):
    prompts = list(prompt_dic.values())
    imgs = list(prompt_dic.keys())
    length_prompts = np.array([len(p) for p in prompts])
    std_len = np.std(length_prompts)
    mean_len = np.mean(length_prompts)
    min_length = max(threshold, mean_len - std_len)
    for img in imgs:
        if len(prompt_dic[img]) < min_length:
            del prompt_dic[img]
    return prompt_dic
def delete_low_related_prompts(img_root_path, prompt_dic):
    prompts = list(prompt_dic.values())
    imgs = list(prompt_dic.keys())
    similarity = {}
    pattern = r"[^a-zA-Z0-9, ]"
    with torch.no_grad():
        for i in range(len(imgs)):
            img_name = imgs[i]
            prompt = prompts[i]
            matches = re.findall(pattern, prompt)
            if len(matches) > 2:
                del prompt_dic[img_name]
            else: 
                img = Image.open(os.path.join(img_root_path, img_name))
                inputs = clip_processor(text=prompt, images=img, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                similarity[img_name] = logits_per_image
    sim = torch.tensor(list((similarity.values())))
    mean_smi = torch.mean(sim)
    std_smi = torch.std(sim)
    threshold = mean_smi - std_smi
    for i in similarity.keys():
        if similarity[i] < threshold:
            del prompt_dic[i]
    print(range(len(imgs)), len(list(prompt_dic.keys())))
    return prompt_dic
def regenerate_prompt(img_root_path, json_data, threshold=30):
    regenerate_prompt = {}
    for img, prompt in json_data.items():
        input = TEMPLATE_V2.format(raw_prompt=prompt)
        input_ids = prompt_tokenizer.encode(input, return_tensors='pt').to(device)
        outputs = prompt_model.generate(
            input_ids,
            max_new_tokens=384,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=5)

        prompts_decode = prompt_tokenizer.batch_decode(outputs[:, input_ids.size(1):], skip_special_tokens=True)
        prompts_decode.append(prompt)
        prompts = [s for s in (max(p.split(","), key=len).strip() for p in prompts_decode) if len(s) < 77]
        if len(prompts) > 0:
            max_prob, bst_prompt = image_txt_sim(img_root_path, [img], prompts)
            regenerate_prompt[img] = bst_prompt
        else:
            continue
    regenerate_prompt = delete_short_prompts(regenerate_prompt, threshold)
    return regenerate_prompt

def delete_img_json(img_root_path, json_data):
    if isinstance(json_data, dict):
        data = json_data
    else:
        data = load(open(json_data), 'r', encoding='utf8')
    img_lst = data.keys()
    for file in os.listdir(img_root_path):
        suffix = file.split(".")[-1]
        if suffix in ["png", "jpeg", "jpg"]:
            if file not in img_lst:
                os.remove(os.path.join(img_root_path, file))
                
def regenerate_part_data(id, data_dir, threshold=30):
    imgs, prompts, work_file = get_part_json(data_dir, id)
    filtered_dic = filter_same_image(work_file, imgs, prompts)
    regenerate_prompts = regenerate_prompt(work_file, filtered_dic)
    regenerate_prompt = delete_low_related_prompts(work_file, regenerate_prompts)
    with open(os.path.join(work_file,f"re-part-{id:06}.json"), 'w') as file:
        json.dump(regenerate_prompt, file)
    print(len(regenerate_prompt.values()))
    delete_img_json(work_file, regenerate_prompt)
    
def label_static(root_img_dir, ids):
    dic = {}
    for id in ids:
        json_keys, prompt, work_file = get_part_json(root_img_dir, id)
        for p in prompt:
            content = max(p.split(","), key=len)
            labels = p.replace(content, "")
            labels = labels.split(",")
            for l in labels:
                if l not in dic.keys():
                    dic[l] = 1
                else:
                    dic[l] += 1
    top_100_labels = sorted(dic.items(), key=lambda item: item[1], reverse=True)
    return dic, top_100_labels

def label_json(root_json_dir, id, labels_dic):
    label_dic = {}
    count = 0
    json_keys, prompt, work_file = get_part_json(root_json_dir, id)
    filtered_dic = filter_same_image(work_file, json_keys, prompt)
    for i in range(len(prompt)):
        p = prompt[i]
        img = json_keys[i]
        label_dic[img] = []
        exist_flag = False
        for word in labels_dic:
            if word in p:
                label_dic[img].append(1)
                exist_flag = True
                count += 1
        if exist_flag == False:
            label_dic[img].append(0)
    print(count)
    return label_dic, work_file
            