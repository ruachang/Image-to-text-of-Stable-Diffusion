export HOME=/home/changl25/
export DATAROOT=/data/changl25

python evaluate.py \
--batch_size 8 \
--save_dir $HOME/Image-to-text-of-Stable-Diffusion/blip2_part1.csv \
--data_root_dir $DATAROOT/Diffusion2DB \
--test_id 1 \
--precision float16 \

python evaluate.py \
--config_path ./configs/VITONHD.yaml \
--batch_size 8 \
--save_dir $HOME/Image-to-text-of-Stable-Diffusion/blip2_flickr30k.csv \
--data_root_dir $DATAROOT/flickr30k \
--precision float16 \
--dataset flickr30k

python evaluate.py \
--model blip2 \
--batch_size 4 \
--save_dir $HOME/Image-to-text-of-Stable-Diffusion/finetune2db2_blip2_test.csv \
--data_root_dir $DATAROOT/Diffusion2DB \
--test_id 1 \
--pretrained \
--model_load_dir $DATAROOT/img2textModel/blip2_model/db_finetune2/3 \
--precision float16
