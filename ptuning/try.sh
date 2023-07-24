PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python try.py \
    --model_name_or_path chatglm-6b \
    --ptuning_checkpoint ../output/checkpoint-128-1e-2/checkpoint-1000 \
    --pre_seq_len $PRE_SEQ_LEN

