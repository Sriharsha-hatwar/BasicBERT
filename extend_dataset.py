import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
import pandas as pd
from tqdm import tqdm
import argparse

class ExtendDataset:
    def __init__(self, file_path):
        self.tsv = pd.read_csv(file_path, sep='\t', header=0)
        self.set_subset_tsv()
    
    def set_subset_tsv(self):
        #self.subset_tsv = self.tsv.iloc[0:10].copy()
        self.subset_tsv_one =  self.tsv.iloc[0:].copy()
        #self.subset_tsv_two = self.tsv.iloc[80100:].copy()
        # create two subsets and create two datasets in the end which can be then merged. 
        #self.subset_tsv.to_csv('data/VUA20/subset.tsv', sep='\t', index=False)


    def replace_word_with_mask(self, text, index, mask_token):
        words = text.split()
        if index >= len(words):
            raise IndexError(f"Index {index} is out of range for the given text.")
        masked_text = ' '.join(words[:index] + [mask_token] + words[index+1:])
        return masked_text
    
    def predict_masked_word(self, masked_text, model, tokenizer):
        input_ids = tokenizer.encode(masked_text, return_tensors='pt')
        mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
        if mask_token_index.size(0) == 0:
            raise ValueError(f"The [MASK] token was not found in the input sequence: {masked_text}")

        mask_token_index = mask_token_index[0]
        output = model(input_ids)
        logits = output.logits
        masked_token_logits = logits[0, mask_token_index, :]
        sorted_logits, sorted_indices = torch.sort(masked_token_logits, descending=True)
        predicted_token_id = sorted_indices[0]
        predicted_token = tokenizer.decode(predicted_token_id)
        return predicted_token

    def replace_word(self, text, index):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        mask_token = tokenizer.mask_token
        masked_text = self.replace_word_with_mask(text, index, mask_token)
        model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()
        predicted_word = self.predict_masked_word(masked_text, model, tokenizer)
        words = masked_text.split()
        words[index] = predicted_word
        result = ' '.join(words)
        return result
    
    def read_from_tsv(self, args):
        tqdm.pandas()
        if args.first:
            print("Starting first partition.")
            self.subset_tsv_one["replaced_sentence"] = self.subset_tsv_one.progress_apply(lambda row: self.replace_word(row['sentence'], row['w_index']), axis=1)
            #print(some_list.tolist())
            #self.subset_tsv["replaced_sentence"] = some_list
            #self.subset_tsv_one.to_csv("data/VUA20/subset-tes-1.tsv", sep="\t", index=False)
            self.subset_tsv_one.to_csv("data/VUA20/new-test.tsv", sep="\t", index=False)
            #print(self.subset_tsv)
        else:
            print("Starting second partition.")
            self.subset_tsv_two["replaced_sentence"] = self.subset_tsv_two.progress_apply(lambda row: self.replace_word(row['sentence'], row['w_index']), axis=1)
            #print(some_list.tolist())
            #self.subset_tsv["replaced_sentence"] = some_list
            self.subset_tsv_two.to_csv("data/VUA20/subset-train-2.tsv", sep="\t", index=False)
            #print(self.subset_tsv)
    
def combine_subsets():
    files = ["data/VUA20/subset-train-1.tsv", "data/VUA20/subset-train-2.tsv"]
    new_file_name = "data/VUA20/new-train.tsv"
    df1 = pd.read_csv(files[0], sep='\t')
    df2 = pd.read_csv(files[1], sep='\t')
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(new_file_name, sep='\t', index=False)
    

if __name__ == "__main__":
    #print("extending dataset.")
    #example_string = "Sleek , solidly built , gentle on the environment , they are often an ideal form of city transport ."
    #index = 5
    obj = ExtendDataset("data/VUA20/test.tsv")
    #new_string = obj.replace_word(example_string, 14)
    #print("new string : ", new_string)
    #print("old string : ", example_string)
    
    parser = argparse.ArgumentParser(description='Extending dataset')
    parser.add_argument('--first', action='store_true')
    args = parser.parse_args()
    obj.read_from_tsv(args)
    #combine_subsets()