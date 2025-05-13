import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# 可根据你模型调整（如果是Flan-T5就用这个）
tokenizer = AutoTokenizer.from_pretrained("/root/bert_response_classifier/bert-base-uncased", local_files_only=True)

def analyze_lengths(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    context_lengths = []
    abstract_lengths = []
    context_token_lens = []
    abstract_token_lens = []

    for item in data:
        context = item["context"]
        abstract = item.get("abstract_30", "")

        context_lengths.append(len(context))
        abstract_lengths.append(len(abstract))

        context_tokens = tokenizer.encode(context, truncation=False)
        abstract_tokens = tokenizer.encode(abstract, truncation=False)

        context_token_lens.append(len(context_tokens))
        abstract_token_lens.append(len(abstract_tokens))

    print("【Context】")
    print(f"  字符：最大 {max(context_lengths)}, 平均 {sum(context_lengths) / len(context_lengths):.2f}")
    print(f"  Tokens：最大 {max(context_token_lens)}, 平均 {sum(context_token_lens) / len(context_token_lens):.2f}")

    print("【Abstract_30】")
    print(f"  字符：最大 {max(abstract_lengths)}, 平均 {sum(abstract_lengths) / len(abstract_lengths):.2f}")
    print(f"  Tokens：最大 {max(abstract_token_lens)}, 平均 {sum(abstract_token_lens) / len(abstract_token_lens):.2f}")

    # 可视化
    plt.figure(figsize=(12, 6))
    plt.hist(context_token_lens, bins=50, alpha=0.6, label="context token length")
    plt.hist(abstract_token_lens, bins=50, alpha=0.6, label="abstract_30 token length")
    plt.axvline(512, color="red", linestyle="--", label="max input = 512")
    plt.legend()
    plt.title("Token Length Distribution")
    plt.xlabel("Token Length")
    plt.ylabel("Sample Count")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    analyze_lengths("./data/Vanilla/train.json")
    # analyze_lengths("./data/narriative/train.json")
    # analyze_lengths("./data/squad/train.json")  # 修改为你的真实路径
