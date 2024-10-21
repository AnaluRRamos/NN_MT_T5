import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from transformers import T5Tokenizer

class T5Dataset(Dataset):
    def __init__(self, data_dir, source_ext, target_ext, tokenizer, max_len=512):
        self.source_files = sorted(glob.glob(os.path.join(data_dir, f"*{source_ext}")))
        self.target_files = sorted(glob.glob(os.path.join(data_dir, f"*{target_ext}")))
        assert len(self.source_files) == len(self.target_files), "Mismatch between source and target files."

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.nlp = spacy.load("en_ner_bionlp13cg_md")

        # Biomedical NE tag mapping
        self.tag_to_idx = {
            'O': 0,  # Outside any named entity
            'AMINO_ACID': 1,
            'ANATOMICAL_SYSTEM': 2,
            'CANCER': 3,
            'CELL': 4,
            'CELLULAR_COMPONENT': 5,
            'DEVELOPING_ANATOMICAL_STRUCTURE': 6,
            'GENE_OR_GENE_PRODUCT': 7,
            'IMMATERIAL_ANATOMICAL_ENTITY': 8,
            'MULTI_TISSUE_STRUCTURE': 9,
            'ORGAN': 10,
            'ORGANISM': 11,
            'ORGANISM_SUBDIVISION': 12,
            'ORGANISM_SUBSTANCE': 13,
            'PATHOLOGICAL_FORMATION': 14,
            'SIMPLE_CHEMICAL': 15,
            'TISSUE': 16,
        }

    def __len__(self):
        return len(self.source_files)

    def load_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def biomedical_ne_tag(self, text):
        doc = self.nlp(text)
        tokens = []
        ne_tags = []
        for token in doc:
            tokens.append(token.text)
            if token.ent_type_:
                ne_tags.append(token.ent_type_)
            else:
                ne_tags.append("O")
        return tokens, ne_tags

    def preprocess(self, text):
        tokens, ne_tags = self.biomedical_ne_tag(text)
        tokenized_text = self.tokenizer(' '.join(tokens), truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)
        aligned_ne_tags = self.align_ne_tags_with_tokens(tokens, ne_tags, input_ids)
        return input_ids, attention_mask, aligned_ne_tags

    def align_ne_tags_with_tokens(self, tokens, ne_tags, input_ids):
        aligned_tags = []
        for word, tag in zip(tokens, ne_tags):
            subwords = self.tokenizer.tokenize(word)
            for _ in subwords:
                aligned_tags.append(tag)
                if len(aligned_tags) >= len(input_ids):
                    break

        if len(aligned_tags) < self.max_len:
            aligned_tags += ['O'] * (self.max_len - len(aligned_tags))

        aligned_ne_tag_ids = torch.tensor([self.tag_to_idx.get(tag, 0) for tag in aligned_tags], dtype=torch.long)
        return aligned_ne_tag_ids

    def __getitem__(self, idx):
        source_text = self.load_file(self.source_files[idx])
        target_text = self.load_file(self.target_files[idx])

        source_input_ids, source_attention_mask, source_ne_tags = self.preprocess(source_text)
        target_input_ids, target_attention_mask, target_ne_tags = self.preprocess(target_text)

        return source_input_ids, source_attention_mask, source_ne_tags, target_input_ids, target_attention_mask, target_ne_tags

def create_dataloaders(data_dir, tokenizer, batch_size, num_workers):
    train_dataset = T5Dataset(data_dir=os.path.join(data_dir, 'train'), tokenizer=tokenizer, source_ext='_en.txt', target_ext='_pt.txt')
    val_dataset = T5Dataset(data_dir=os.path.join(data_dir, 'val'), tokenizer=tokenizer, source_ext='_en.txt', target_ext='_pt.txt')
    test_dataset = T5Dataset(data_dir=os.path.join(data_dir, 'test'), tokenizer=tokenizer, source_ext='_en.txt', target_ext='_pt.txt')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader
