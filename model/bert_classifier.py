# model/bert_classifier.py
import torch
import torch.nn as nn
from transformers import BertModel
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import wandb


class BertClassifierLightning(pl.LightningModule):
    def __init__(self, model_name, num_labels=5, lr=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_output)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
