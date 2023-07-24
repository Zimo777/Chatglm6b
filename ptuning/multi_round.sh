#P-tuning v2 微调
PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python multi_round.py \
    --do_train \
    --train_file make_dataset/answers1.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path ..\\chatglm-6b \
    --output_dir output/school-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 2e-2 \
    --pre_seq_len 128 \
    --quantization_bit 8

