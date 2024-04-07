# Image-to-text-of-Stable-Diffusion

## For Reference

* Blip (https://github.com/salesforce/BLIP)
* Project/model released after Blip (https://github.com/salesforce/lavis)

## Some techniques?

* LoRA for finetune
    * basic version
        - [x] achieve
        - [ ] load lora model: can not load from config
        - [ ] change the dataloader to load more data
        - [ ] some data augmentation
        - [ ] train parameters adjustment: add another loss function?
    * quantized version
* Model ensemble
* More dataset/combined dataset