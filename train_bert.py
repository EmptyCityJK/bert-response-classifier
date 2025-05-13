# train_bert.py
import torch
import os
from datetime import datetime
from model.bert_classifier import BertClassifierLightning
from transformers import AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from preprocess.dataset import load_dataset
import wandb
import argparse

torch.set_float32_matmul_precision('high')

def train_bert(model_name, num_labels=5, lr=2e-5, max_epochs=3, gpus=1, batch_size=32, num_workers=4, data_dir='./data', resume_checkpoint=None):
    # 初始化 tokenizer 和数据集
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    train_loader, val_loader = load_dataset(tokenizer, data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)

    # 初始化模型
    model = BertClassifierLightning(model_name=model_name, num_labels=num_labels, lr=lr)

    # 创建 checkpoints 文件夹
    os.makedirs("checkpoints", exist_ok=True)
    
    # 初始化模型
    if resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        print(f"Loading model from checkpoint: {resume_checkpoint}")
        model = BertClassifierLightning.load_from_checkpoint(resume_checkpoint, model_name=model_name, num_labels=num_labels, lr=lr)
    else:
        print("No checkpoint provided or not found. Training from scratch.")
        model = BertClassifierLightning(model_name=model_name, num_labels=num_labels, lr=lr)
    
    # 获取当前时间戳用于命名
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 提取数据集文件夹名称
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    
    # 定义 checkpoint 保存逻辑
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"bert-{dataset_name}-acc={{val/acc:.4f}}-{current_time}",
        monitor="val/acc",
        mode="max",  # 保存验证准确率最高的模型
        save_top_k=1,
        save_weights_only=True,
    )
    
    # 初始化Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=gpus,
        precision="16-mixed",
        # log_every_n_steps=10,
        logger=pl.loggers.WandbLogger(project='bert-classifier'),
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="BERT Model Training")
    parser.add_argument('--model_name', type=str, default="/root/bert_response_classifier/bert-base-uncased")
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='./data/Vanilla')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_epochs', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    # Train the model with the given arguments
    train_bert(
        model_name=args.model_name,
        num_labels=args.num_labels,
        lr=args.lr,
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        resume_checkpoint=args.checkpoint
    )