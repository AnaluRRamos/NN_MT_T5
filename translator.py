import config
from models import T5FineTuner
import torch
from dataset import create_adapted_tokenizer

class PredictModel:
    def __init__(self, resume_from_checkpoint, max_length=256):
        model_name = config.get_model_name()
        self.tokenizer, self.added_tokens = create_adapted_tokenizer(model_name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = T5FineTuner.load_from_checkpoint(resume_from_checkpoint, tokenizer=self.tokenizer,
                                                      train_dataloader=None, val_dataloader=None,
                                                      test_dataloader=None, learning_rate=1e-5,
                                                      added_tokens=self.added_tokens, target_max_length=max_length).to(self.device)

    def predict_en_pt(self, text):
        max_length = config.get_source_max_length()

        sent = "translate English to Portuguese: " + text + self.tokenizer.eos_token
        tok = self.tokenizer.encode_plus(sent, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=max_length, padding='max_length')
        pred = self.model(tok['input_ids'].to(self.device), tok['attention_mask'].to(self.device))

        sys = [self.tokenizer.decode(tokens) for tokens in pred]
        return sys

    def predict_batch_en_pt(self, text_list):
        max_length = config.get_source_max_length()
        sent_list = ["translate English to Portuguese: " + text + self.tokenizer.eos_token for text in text_list]

        tok = self.tokenizer.batch_encode_plus(sent_list, return_tensors='pt', truncation=True, add_special_tokens=True, max_length=max_length, padding='max_length')
        pred = self.model(tok['input_ids'].to(self.device), tok['attention_mask'].to(self.device))

        sys = [self.tokenizer.decode(tokens) for tokens in pred]
        return sys

    def predict_pt_en(self, text):
        max_length = config.get_source_max_length()

        sent = "translate Portuguese to English: " + text + self.tokenizer.eos_token
        tok = self.tokenizer.encode_plus(sent, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=max_length, padding='max_length')
        pred = self.model(tok['input_ids'].to(self.device), tok['attention_mask'].to(self.device))

        sys = [self.tokenizer.decode(tokens) for tokens in pred]
        return sys

    def predict_batch_pt_en(self, text_list):
        max_length = config.get_source_max_length()
        sent_list = ["translate Portuguese to English: " + text + self.tokenizer.eos_token for text in text_list]

        tok = self.tokenizer.batch_encode_plus(sent_list, return_tensors='pt', truncation=True, add_special_tokens=True, max_length=max_length, padding='max_length')
        pred = self.model(tok['input_ids'].to(self.device), tok['attention_mask'].to(self.device))

        sys = [self.tokenizer.decode(tokens) for tokens in pred]
        return sys



