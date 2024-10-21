import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import T5Tokenizer
from dataset import create_dataloaders
from model import create_model
import argparse
import torch
import json

def main(args):
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)
    
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = create_model(
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=args.learning_rate
    )

    train_metrics = {'train_loss': [], 'val_loss': [], 'val_bleu': []}

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[ModelCheckpoint(monitor='val_bleu')],
        precision=16 if torch.cuda.is_available() else 32,
        gpus=1 if torch.cuda.is_available() else 0,
        logger=False
    )

    trainer.fit(model)

    # After training, save the logged metrics
    with open('train_val_metrics.json', 'w') as f:
        json.dump(train_metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5 Fine-Tuning Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--tokenizer_name", type=str, default="t5-small", help="Name of the T5 tokenizer (default: t5-small).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader (default: 8).")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for training (default: 3e-5).")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of epochs for training (default: 5).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader (default: 4).")

    args = parser.parse_args()
    main(args)


