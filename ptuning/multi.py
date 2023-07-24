import streamlit as st
import torch
import os, sys
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
import base64

model_path = 'chatglm-6b'
ptuning_checkpoint = './output/school6-128-2e-2/checkpoint-3000'

# 超参数
max_length = 4096
top_p = 0.85
temperature = 0.95


def get_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.pre_seq_len = 128
    config.prefix_projection = False

    if ptuning_checkpoint is not None:
        print(f"Loading prefix_encoder weight from {ptuning_checkpoint}")
        model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
    if config.pre_seq_len is not None:
        # P-tuning v2
        model = model.quantize(8).half().cuda()
        model.transformer.prefix_encoder.float().cuda()

    model = model.quantize(8).half().cuda()
    model.transformer.prefix_encoder.float().cuda()
    model = model.eval()
    return tokenizer, model


def predict(input, max_length, top_p, temperature, history=None):
    tokenizer, model = get_model()
    print('AI正在回复:')
    history = []
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        query, response = history[-1]
        print(response)
    return history


if __name__ == '__main__':
    while True:
        question = input('请输入你想问的问题:')
        history = predict(question, max_length, top_p, temperature)
