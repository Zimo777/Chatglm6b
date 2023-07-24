import streamlit as st
from streamlit_chat import message
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

# python -m streamlit run ./web_test.py --server.port 27777 --server.address 0.0.0.0

model_path = 'chatglm-6b'
ptuning_checkpoint = './output/school6-128-2e-2/checkpoint-3000'
st.set_page_config(
    page_title="城理AI助手",
    page_icon=":robot:"
)

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


@st.cache_resource
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

    with container:
        message(input, avatar_style="big-smile", key=str(len(history)) + "_user")
        st.write("AI正在回复:")
        with st.empty():
            history = []
            for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                                       temperature=temperature):
                query, response = history[-1]
                st.write(response)
            message(response, avatar_style="bottts", key=str(len(response)))
    return history


if __name__ == '__main__':

    container = st.container()

    prompt_text = st.text_area(label="问题输入",
                               height=100,
                               placeholder="请在这儿输入您的问题")

    max_length = st.sidebar.slider(
        'max_length', 0, 4096, 2048, step=1
    )
    top_p = st.sidebar.slider(
        'top_p', 0.0, 1.0, 0.85, step=0.01
    )
    temperature = st.sidebar.slider(
        'temperature', 0.0, 1.0, 0.95, step=0.01
    )

    if 'state' not in st.session_state:
        st.session_state['state'] = []

    if st.button("发送", key="predict"):
        with st.spinner("城理AI助手正在思考，请稍等........"):
            # text generation
            st.session_state["state"] = predict(prompt_text, max_length, top_p, temperature, st.session_state["state"])
