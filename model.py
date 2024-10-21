import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sacrebleu

class T5FineTuner(pl.LightningModule):
    def __init__(self, tokenizer, train_dataloader, val_dataloader, test_dataloader, learning_rate, target_max_length=32):
        super(T5FineTuner, self).__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader
        self.model = T5ForConditionalGeneration.from_pretrained(tokenizer.name_or_path)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.target_max_length = target_max_length

    def forward(self, source_token_ids, source_mask, target_token_ids=None, target_mask=None, ne_tag_mask=None, training=False):
        if training:
            target_token_ids[target_token_ids == self.tokenizer.pad_token_id] = -100
            output = self.model(input_ids=source_token_ids, attention_mask=source_mask, labels=target_token_ids)
            if ne_tag_mask is not None:
                attention_weights = self.model.encoder.last_hidden_state
                ne_focus_loss = self.calculate_ne_focus_loss(attention_weights, ne_tag_mask)
                loss = output.loss + ne_focus_loss
            else:
                loss = output.loss
            return loss
        else:
            predicted_token_ids = self.model.generate(input_ids=source_token_ids, max_length=self.target_max_length)
            return predicted_token_ids

    def calculate_ne_focus_loss(self, attention_weights, ne_tag_mask):
        avg_attention_weights = attention_weights.mean(dim=-1)
        ne_attention = avg_attention_weights * ne_tag_mask
        ne_focus_loss = torch.mean(1.0 - ne_attention)
        return ne_focus_loss

    def training_step(self, batch, batch_idx):
        source_token_ids, source_mask, target_token_ids, target_mask, source_ne_tags, target_ne_tags = batch
        loss = self(source_token_ids, source_mask, target_token_ids, target_mask, training=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source_token_ids, source_mask, target_token_ids, target_mask, source_ne_tags, target_ne_tags = batch
        val_loss = self(source_token_ids, source_mask, target_token_ids, target_mask, training=True)
        
        pred_token_ids = self(source_token_ids, source_mask)
        pred_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_token_ids]
        target_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in target_token_ids]
        
        bleu_score = sacrebleu.corpus_bleu(pred_texts, [target_texts]).score
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_bleu', bleu_score, prog_bar=True)
        
        return val_loss

    def on_validation_epoch_end(self, outputs):
        trues = sum([list(x['target']) for x in outputs], [])
        preds = sum([list(x['pred']) for x in outputs], [])
        bleu = sacrebleu.corpus_bleu(trues, [preds]).score
        self.log('val_bleu', bleu, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.learning_rate)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader


def create_model(tokenizer, train_dataloader, val_dataloader, test_dataloader, learning_rate, target_max_length=32):
    model = T5FineTuner(
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        learning_rate=learning_rate,
        target_max_length=target_max_length
    )
    return model
