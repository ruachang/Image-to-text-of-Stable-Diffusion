export HOME=/home/changl25/
export DATAROOT=/data/changl25

# script for first stage fine-tuning on Flickr30k
python blip_finetune.py \
--config_path ./configs/VITONHD.yaml \
--batch_size 4 \
--epochs 5 \
--lora_dropout 0.05 \
--lora_rank 2 \
--lora_alpha 32 \
--lora_module language_projection query key value \
--lora_bias "none" \
--model_save_dir $DATAROOT/img2textModel/blip2_model/flickr1k \
--save_dir $HOME/Image-to-text-of-Stable-Diffusion/finetune_blip2_flickr3.csv \
--data_root_dir $DATAROOT/flickr30k \
--dataset flickr30k \
--split 25 \
--train_id 2 \
--test_id 1 \
--warmup \
--similar_loss \
--precision float16 \

# script for second stage fine-tuning on DiffusionDB with ssl
python blip_finetune.py \
--batch_size 4 \
--epochs 5 \
--lora_dropout 0.05 \
--lora_rank 2 \
--lora_alpha 32 \
--lora_module language_projection query key value qkv\
--lora_bias "none" \
--model_save_dir $DATAROOT/img2textModel/blip2_model/db_finetune2 \
--save_dir $HOME/Image-to-text-of-Stable-Diffusion/finetune2db2_blip2_test.csv \
--data_root_dir $DATAROOT/Diffusion2DB \
--train_id 2 3 4 \
--test_id 1 \
--warmup \
--similar_loss \
--precision float16 \
--pretrained \
--model_load_dir $DATAROOT/img2textModel/blip2_model/flickr6k/0 \
--wandb
