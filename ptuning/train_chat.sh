#多轮对话微调
PRE_SEQ_LEN=128
LR=1e-2

CUDA_VISIBLE_DEVICES=0  main.py \
    --do_train \
    --train_file make_dataset/answers.json \
    --prompt_column prompt \
    --response_column response \
    --history_column history \
    --overwrite_cache \
    --model_name_or_path chatglm-6b \
    --output_dir ../output/checkpoint-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8

