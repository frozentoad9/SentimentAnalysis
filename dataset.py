import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig

class SSTDataset(Dataset):

	def __init__(self, filename, maxlen, tokenizer):

		#This is the file where sentences and labels are stored (its a .tsv format file)
		self.df = pd.read_csv(filename, delimiter='\t')
		#Initialize the tokenizer for the desired transformer model
		self.tokenizer = tokenizer
		#Maximum length of the tokens list to keep all the sequences of fixed size
		self.maxlen = maxlen

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		#Select the sentence and label at the specified index in the data frame
		#sentence = self.df.loc[index, 'sentence']
		sentence = self.df.loc[index, 'content']
		label = self.df.loc[index, 'label']
		#Preprocess the text to be suitable for the transformer
		input_ids = self.tokenizer.encode_plus( sentence,
		 					add_special_tokens=True,
							return_attention_mask=True,
							truncation=True,
							max_length=self.maxlen,
							padding='max_length',
							return_tensors='pt')

		return input_ids['input_ids'], input_ids['attention_mask'], label
