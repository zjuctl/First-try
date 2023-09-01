import numpy as np
import pandas as pd
from mindspore.dataset import GeneratorDataset
from mindspore import nn
from mindnlp.models import GPTForSequenceClassification
from mindnlp._legacy.amp import auto_mixed_precision
from mindnlp.transforms import GPTTokenizer
from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy

std = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

file_path = './mbti_1.csv'  # 数据路径

df = pd.read_csv(file_path)

type = df['type'].tolist()
posts = df['posts'].tolist()

label2idx = {label: i for i, label in enumerate(std)}

type = [label2idx[sent] for sent in type]
type = np.array(type)

def get_singlelabel_data(type, posts):
    """定义生成单标签数据集函数"""
    for i in range(len(type)):
        data = posts[i]  # 自定义数据
        label = type[i]  # 自定义标签
        yield np.array([data]).astype(str), np.array(label).astype(np.int32)

# 定义数据集
dataset = GeneratorDataset(list(get_singlelabel_data(type, posts)), column_names=['text', 'label'])

def process_dataset(dataset, tokenizer, max_seq_len=256, batch_size=32, shuffle=False):
    """数据集预处理"""
    def pad_sample(text):
        if len(text) + 2 >= max_seq_len:
            return np.concatenate(
                [np.array([tokenizer.bos_token_id]), text[: max_seq_len-2], np.array([tokenizer.eos_token_id])]
            )
        else:
            pad_len = max_seq_len - len(text) - 2
            return np.concatenate(
                [np.array([tokenizer.bos_token_id]), text,
                 np.array([tokenizer.eos_token_id]),
                 np.array([tokenizer.pad_token_id] * pad_len)]
            )

    column_names = ["text", "label"]
    rename_columns = ["input_ids", "label"]

    if shuffle:
        dataset = dataset.shuffle(batch_size)

    # map dataset
    dataset = dataset.map(operations=[tokenizer, pad_sample], input_columns="text")
    # rename dataset
    dataset = dataset.rename(input_columns=column_names, output_columns=rename_columns)
    # batch dataset
    dataset = dataset.batch(batch_size)

    return dataset

# tokenizer
gpt_tokenizer = GPTTokenizer.from_pretrained('openai-gpt')

# add sepcial token: <bos> <eos> <pad>
special_tokens_dict = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
}
num_added_toks = gpt_tokenizer.add_special_tokens(special_tokens_dict)

# split train dataset into train and valid datasets
train, val, test = dataset.split([0.7, 0.1, 0.2])

dataset_train = process_dataset(train, gpt_tokenizer, shuffle=True)
dataset_val = process_dataset(val, gpt_tokenizer)
dataset_test = process_dataset(test, gpt_tokenizer)

# set bert config and define parameters for training
model = GPTForSequenceClassification.from_pretrained('openai-gpt', num_labels=16)
model.pad_token_id = gpt_tokenizer.pad_token_id
model.resize_token_embeddings(model.config.vocab_size + 3)
model = auto_mixed_precision(model, 'O1')   # “O1” - 将白名单内的Cell和运算转为float16精度，其余部分保持float32精度。

loss = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=2e-5)

metric = Accuracy()

# define callbacks to save checkpoints
ckpoint_cb = CheckpointCallback(save_path='checkpoint', ckpt_name='sentiment_model', epochs=1, keep_checkpoint_max=2)
best_model_cb = BestModelCallback(save_path='checkpoint', auto_load=True)

trainer = Trainer(network=model, train_dataset=dataset_train,
                  eval_dataset=dataset_val, metrics=metric,
                  epochs=30, loss_fn=loss, optimizer=optimizer, callbacks=[ckpoint_cb, best_model_cb],
                  jit=True)

trainer.run(tgt_columns="label")

evaluator = Evaluator(network=model, eval_dataset=dataset_test, metrics=metric)
evaluator.run(tgt_columns="label")