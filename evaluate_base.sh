export HOME=/home/changl25/
export DATAROOT=/data/changl25

# python evaluate.py \
# --config_path ./configs/VITONHD.yaml \
# --batch_size 8 \
# --save_dir $HOME/Image-to-text-of-Stable-Diffusion/blip2_part1.csv \
# --data_root_dir $DATAROOT/Diffusion2DB \
# --test_id 1 \
# --precision float16 \

# python evaluate.py \
# --config_path ./configs/VITONHD.yaml \
# --batch_size 8 \
# --save_dir $HOME/Image-to-text-of-Stable-Diffusion/blip2_flickr30k.csv \
# --data_root_dir $DATAROOT/flickr30k \
# --precision float16 \
# --dataset flickr30k

# python evaluate.py \
# --config_path ./configs/VITONHD.yaml \
# --model blip \
# --batch_size 8 \
# --save_dir $HOME/Image-to-text-of-Stable-Diffusion/blip_part1.csv \
# --data_root_dir $DATAROOT/Diffusion2DB \
# --test_id 1 \

# python evaluate.py \
# --config_path ./configs/VITONHD.yaml \
# --model blip \
# --batch_size 8 \
# --save_dir $HOME/Image-to-text-of-Stable-Diffusion/blip_flickr30k.csv \
# --data_root_dir $DATAROOT/flickr30k \
# --dataset flickr30k

python evaluate.py \
--config_path ./configs/VITONHD.yaml \
--model blip \
--batch_size 8 \
--save_dir $HOME/Image-to-text-of-Stable-Diffusion/blip_part1_finetune_e.csv \
--data_root_dir $DATAROOT/Diffusion2DB \
--test_id 1 \
--pretrained \
--model_load_dir $DATAROOT/img2textModel/blip_model_edit/best