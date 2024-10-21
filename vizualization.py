import matplotlib.pyplot as plt
import os
import json
import torch
import numpy as np
from transformers import T5Tokenizer
import pytorch_lightning as pl
from dataset import create_dataloaders
from model import create_model

def plot_loss(train_losses, val_losses, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.show()

def plot_bleu_score(bleu_scores, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, bleu_scores, label='Validation BLEU Score')
    plt.xlabel('Epochs')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score Over Epochs')
    plt.legend()
    plt.show()

def plot_attention_weights(model, tokenizer, text):
    model.eval()
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, output_attentions=True)
    
    attention = outputs.decoder_attentions[-1].squeeze().cpu().numpy()

    plt.matshow(attention)
    plt.title('Attention Weights')
    plt.colorbar()
    plt.show()

def compare_predictions(model, tokenizer, dataloader, num_examples=5):
    model.eval()
    tokenizer.decode
    for batch in dataloader:
        source_token_ids, source_mask, target_token_ids, target_mask, source_ne_tags, target_ne_tags = batch
        with torch.no_grad():
            pred_token_ids = model.generate(input_ids=source_token_ids, max_length=32)
        source_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in source_token_ids]
        target_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in target_token_ids]
        pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_token_ids]

        for i in range(min(num_examples, len(source_texts))):
            print(f"Source: {source_texts[i]}")
            print(f"Target: {target_texts[i]}")
            print(f"Prediction: {pred_texts[i]}")
            print()

def main():
    # Define paths
    model_checkpoint = 'checkpoints/your_model_checkpoint.ckpt'
    data_dir = 'data/'
    
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    _, val_dataloader, _ = create_dataloaders(
        data_dir=data_dir, tokenizer=tokenizer, batch_size=8, num_workers=4
    )
    model = create_model(
        tokenizer=tokenizer,
        train_dataloader=None,
        val_dataloader=val_dataloader,
        test_dataloader=None,
        learning_rate=3e-5
    )
    model = model.load_from_checkpoint(checkpoint_path=model_checkpoint)
    
    # Assuming you have saved losses and BLEU scores in a file during training
    with open('train_val_metrics.json', 'r') as f:
        metrics = json.load(f)

    train_losses = metrics['train_loss']
    val_losses = metrics['val_loss']
    bleu_scores = metrics['val_bleu']
    epochs = list(range(1, len(train_losses) + 1))

    plot_loss(train_losses, val_losses, epochs)
    plot_bleu_score(bleu_scores, epochs)
    plot_attention_weights(model, tokenizer, "Translate this sentence from English to Portuguese.")
    compare_predictions(model, tokenizer, val_dataloader)

if __name__ == "__main__":
    main()
