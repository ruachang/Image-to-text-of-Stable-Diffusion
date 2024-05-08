export HOME=/home/changl25/
export DATAROOT=/data/changl25

# python blip_finetune.py \
# --config_path ./configs/VITONHD.yaml \
# --batch_size 4 \
# --epochs 5 \
# --lora_dropout 0.05 \
# --lora_rank 2 \
# --lora_alpha 32 \
# --lora_module language_projection query key value \
# --lora_bias "none" \
# --model_save_dir $DATAROOT/img2textModel/blip2_model \
# --save_dir $HOME/Image-to-text-of-Stable-Diffusion/finetune_blip2_test.csv \
# --data_root_dir $DATAROOT/Diffusion2DB \
# --train_id 2 \
# --test_id 1 \
# --warmup \
# --precision float16 \
# --wandb


python blip_finetune.py \
--config_path ./configs/VITONHD.yaml \
--batch_size 4 \
--epochs 5 \
--lora_dropout 0.05 \
--lora_rank 2 \
--lora_alpha 32 \
--lora_module language_projection query key value \
--lora_bias "none" \
--model_save_dir $DATAROOT/img2textModel/blip2_model/flickr \
--save_dir $HOME/Image-to-text-of-Stable-Diffusion/finetune_blip2_flickr.csv \
--data_root_dir $DATAROOT/flickr30k \
--dataset flickr30k \
--split 5 \
--train_id 2 \
--test_id 1 \
--warmup \
--precision float16 \
--wandb