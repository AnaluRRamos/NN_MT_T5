from dataset import NMTDataset

data_dir = "data/train/"

dataset = NMTDataset(data_dir=data_dir, source_ext="_en.txt", target_ext="_pt.txt", tokenizer_name="t5-small", max_len=100)

sample = dataset[0]

print("Source Input IDs:", sample[0])
print("Source Attention Mask:", sample[1])
print("Source NE Tags:", sample[2])
print("Target Input IDs:", sample[3])
print("Target Attention Mask:", sample[4])
print("Target NE Tags:", sample[5])

