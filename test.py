import pytorch_lightning as pl
from transformers import T5Tokenizer
from dataset import create_dataloaders
from model import create_model
import argparse
import torch

def main(args):
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)

    _, _, test_dataloader = create_dataloaders(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = create_model(
        tokenizer=tokenizer,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=test_dataloader,
        learning_rate=args.learning_rate
    )
    model = model.load_from_checkpoint(checkpoint_path=args.checkpoint_path)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0
    )

    trainer.test(model, test_dataloaders=test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Script for Trained Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--tokenizer_name", type=str, default="t5-small", help="Name of the T5 tokenizer (default: t5-small).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader (default: 8).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader (default: 4).")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate (default: 3e-5).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")

    args = parser.parse_args()
    main(args)
