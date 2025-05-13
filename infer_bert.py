import torch
import json
import os
from transformers import AutoTokenizer
from model.bert_classifier import BertClassifierLightning
from tqdm import tqdm

def load_test_data(test_path):
    """Load test data which contains only one response field per item"""
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_items = []
    for item in data:
        context = item["context"]
        abstract = item.get("abstract_30", "")
        qas = item["qas"]
        student_answer = item["response"]  # Only one response field
        
        # Build QA pairs
        qa_pairs = []
        for qa, ans in zip(qas, student_answer):
            question = qa.get("question", "没有问题")
            qa_pairs.append(f"{question}：{ans}")
        qa_section = " [SEP] ".join(qa_pairs)
        
        # Build full input text
        input_text = f"{qa_section} [SEP] {abstract} [SEP] {context}"
        
        test_items.append({
            "text": input_text,
            "id": item.get("id", len(test_items))  # Use index as ID if not provided
        })
    
    return test_items

def predict(model, tokenizer, test_items, max_length=512, batch_size=32):
    """Perform batch prediction on test items"""
    model.eval()
    predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_items), batch_size), desc="Predicting"):
        batch_items = test_items[i:i+batch_size]
        
        # Tokenize batch
        texts = [item["text"] for item in batch_items]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        
        # Predict
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Map predictions to labels
        label_map = {
            0: "fully_response",
            1: "partially_response",
            2: "blank_response"
        }
        for pred in batch_preds:
            predictions.append(label_map[pred])
    
    return predictions

def save_predictions(predictions, output_file):
    """Save predictions to a text file"""
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == "__main__":
    # Configuration
    checkpoint_filename = "acc=0.9467-20250513-193203.ckpt"  # Just the filename
    model_path = os.path.join("checkpoints/bert-narriative-acc=val", checkpoint_filename)  # Proper path construction
    model_name = "/root/bert_response_classifier/bert-base-uncased"
    test_path = "./data/narriative/test.json"  # Path to your test file
    output_file = "predictions.txt"  # Output file name
    
    # Check if checkpoint exists
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint file not found at {model_path}")
        print("Available checkpoints:")
        for f in os.listdir("checkpoints"):
            print(f"  - {f}")
        exit(1)
    
    # Load model
    print("Loading model...")
    model = BertClassifierLightning.load_from_checkpoint(
        model_path,
        model_name=model_name,
        num_labels=3
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    
    # Load and prepare test data
    print("Loading test data...")
    test_items = load_test_data(test_path)
    
    # Perform prediction
    print("Making predictions...")
    predictions = predict(model, tokenizer, test_items)
    
    # Save results
    save_predictions(predictions, output_file)
    print(f"Predictions saved to {output_file}")