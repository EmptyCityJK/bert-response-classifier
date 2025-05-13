import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_and_flatten(data_path)

    def load_and_flatten(self, path):
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        label_map = {
            "fully_response": 0,
            "partially_response": 1,
            "blank_response": 2
        }

        flattened = []

        for item in raw_data:
            context = item["context"]
            abstract = item.get("abstract_30", "")
            qas = item["qas"]

            for key in label_map:
                if key not in item:
                    continue

                student_answers = item[key]
                if not student_answers or len(student_answers) != len(qas):
                    continue  # 保证QA数量一致且不为空

                # 构建 QA 拼接段
                qa_pairs = []
                for qa, ans in zip(qas, student_answers):
                    question = qa.get("question", "没有问题")
                    qa_pairs.append(f"{question}：{ans}")
                qa_section = " [SEP] ".join(qa_pairs)

                # 构建完整输入
                input_text = f"{qa_section} [SEP] {abstract} [SEP] {context}"

                # 打印第一个样本进行调试
                if not flattened:
                    print("First input_text:", input_text)

                flattened.append({
                    "text": input_text,
                    "label": label_map[key]
                })

        return flattened

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }


def load_dataset(tokenizer, data_dir, max_length=512, batch_size=32, num_workers=8):
    from torch.utils.data import DataLoader

    train_set = TextDataset(f"{data_dir}/train.json", tokenizer, max_length)
    val_set = TextDataset(f"{data_dir}/valid.json", tokenizer, max_length)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
