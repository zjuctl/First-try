import translate
import mindspore
import numpy as np
from mindnlp.models import GPTForSequenceClassification
from mindnlp.transforms import GPTTokenizer
from mindnlp._legacy.amp import auto_mixed_precision

std = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
special_tokens_dict = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
}
gpt_tokenizer = GPTTokenizer.from_pretrained('openai-gpt')
num_added_toks = gpt_tokenizer.add_special_tokens(special_tokens_dict)
max_seq_len = 256


def pad_sample(text):
    if len(text) + 2 >= max_seq_len:
        return np.concatenate(
            [np.array([gpt_tokenizer.bos_token_id]), text[: max_seq_len - 2], np.array([gpt_tokenizer.eos_token_id])]
        )
    else:
        pad_len = max_seq_len - len(text) - 2
        return np.concatenate(
            [np.array([gpt_tokenizer.bos_token_id]), text,
             np.array([gpt_tokenizer.eos_token_id]),
             np.array([gpt_tokenizer.pad_token_id] * pad_len)]
        )


# Instantiate a random initialized model
model = GPTForSequenceClassification.from_pretrained('openai-gpt', num_labels=16)
model.pad_token_id = gpt_tokenizer.pad_token_id
model.resize_token_embeddings(model.config.vocab_size + 3)
model = auto_mixed_precision(model, 'O1')   # “O1” - 将白名单内的Cell和运算转为float16精度，其余部分保持float32精度。

params = mindspore.load_checkpoint("./checkpoint/best_so_far.ckpt") #
unsec = mindspore.load_param_into_net(model, params)

model.set_train(False)
idx2label = {i: labels for i, labels in enumerate(std)}
language = input("Please select the language | 请选择语言  0) English|英语  1) Chinese|汉语 : ")

if language == "0":
    data = input("Please input the posts:")
else:
    data = input("请输入用于检测的文本信息: ")
    data = translate.trans(data)

data = gpt_tokenizer(data)
data = pad_sample(data)

# print(data.dtype)
data = mindspore.Tensor(data).expand_dims(axis=0)
# (logits, attn_score, ..)
data = model(data)[0]
data = np.array(data).argmax()
pred = idx2label[data]
print(pred)
# [batch_size, seq_len]