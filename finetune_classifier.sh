export HOME=/home/changl25/
export DATAROOT=/data/changl25

options=("real" "unreal" "fantasy" "detailed" "res" "focus" "cinematic" "painting" "digital")


for i in "${!options[@]}"; do
#   printf "%d) %s\n" $((i+1)) "${options[$i]}"
  python classifier_train.py \
  --batch_size 4 \
  --epochs 3 \
  --data_root_dir $DATAROOT/Diffusion2DB_label \
  --label ${options[$i]} \
  --train_id 2 3 \
  --test_id 1 \
  --pretrained \
  --model_load_dir $DATAROOT/img2textModel/blip2_model/db_finetune2/best \
  --precision float16 \
  --warmup \
  --model_save_dir $DATAROOT/img2textModel/classifier_finetune
done

python classifier_train.py \
  --batch_size 4 \
  --data_root_dir $DATAROOT/Diffusion2DB_label \
  --test_id 1 \
  --precision float16 \
  --classifier_load_dir $DATAROOT/img2textModel/classifier_finetune/best \
  --test \
  --test_label real unreal fantasy detailed res focus cinematic painting digital \
  --save_dir $HOME/Image-to-text-of-Stable-Diffusion/classifier_fintune.csv
  
python evaluate_final.py \
--data_root_dir $DATAROOT/Diffusion2DB_label \
--test_id 1 \
--classifier_load_dir $DATAROOT/img2textModel/classifier/best \
--keyword fantasy focus digital detailed res cinematic painting \
--save_dir $HOME/Image-to-text-of-Stable-Diffusion/prompt_generate.csv
