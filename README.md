# Image-to-text-of-Stable-Diffusion

This repo is the code for project Staged-finetune of image-to-prompt synthesis. The code includes

* Finetuning code and script for caption model: 
    * code: `blip_finetune.py`
    * script: `script/finetune_base.sh`

* Training code and script for style classification:
    * code: `classifier_train.py`
    * script: `script/finetune_classifier.sh`

* Evaluate code and script for evaluation and generation
    * Evaluation: `evaluate.py`
    * Generation: `evaluate_final.py`
    * script: `script/evaluate_base.sh`

* Reformate and generate dataset
    * Step by step ver: `regenerate_prompt.ipynb`
    * Related function `prompt_edit.py`

Some example generated prompts are in `result\`

