import torch
from peft import LoraConfig, get_peft_model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
def get_lora_model(model, configs):
    lora_rank = configs["lora_rank"]
    lora_alpha = configs["lora_alpha"]
    lora_bias = configs["lora_bias"]
    lora_dropout   = configs["lora_dropout"]
    lora_target_modules = configs["target_modules"]
    lora_config = LoraConfig(
        r              = lora_rank,
        lora_alpha     = lora_alpha, 
        lora_dropout   = lora_dropout,
        bias           = lora_bias,
        target_modules = lora_target_modules
    )
    
    peft_model = get_peft_model(model, lora_config)
    return peft_model

def print_cuda_memory_statistics(device):
    print(f"Total memory: {torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)} MB")
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f"Allocated memory: {allocated_memory} MB")
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"Reserved memory: {reserved_memory} MB")
    print(f"Free memory within reserved: {reserved_memory - allocated_memory} MB")
