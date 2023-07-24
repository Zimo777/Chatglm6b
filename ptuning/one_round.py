import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

CHECKPOINT_PATH = 'output/school5-128-2e-2/checkpoint-2000'
model_path="chatglm-6b"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
# 此处使用你的 ptuning 工作目录
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.quantize(8)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

while True:
    print("正在使用城理小助手")
    dialogue=input("请输入你要问答的问题：")
    response, history = model.chat(tokenizer, dialogue, history=[])
    print(response)
    if dialogue=='结束':
        break