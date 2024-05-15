import sys
import csv
import pandas as pd
#Drop columns: https://www.geeksforgeeks.org/delete-a-csv-column-in-python/
"""Simplifies VUA20 for easier use + writes words to file with a metaphorical meaning but no basic meaning
"""
def remove_columns(dataset_name, modified_dataset):
    """Removes columns not necessary for our work, then gets relevant word based on the index
    writes result to modified_dataset"""
    weird_char = [';', ':', '!', "*", "]", "["]
    with open(dataset_name, "r") as input: 
        reader = csv.reader(input, delimiter="\t") 

        with open(modified_dataset, "w") as output: 
            writer = csv.writer(output, delimiter="\t") 
            row = next(reader)

            for row in reader:
                index = int(row[5])
                word = row[2].split(" ")[index].lower()

                for i in weird_char:
                    word = word.replace(i, '')

                writer.writerow((row[1], word))
    return 

def count_metaphor_occurances(modified_dataset_name, counts_of_metaphors):
    """Removes columns not necessary for our work, writes result to modified_data_set
    name
    - Metaphor_bool     Word
    """
    #metaphor_dict = {word : [count metaphors, count basic]}
    metaphor_dict = {}
    with open(modified_dataset_name, "r") as input: 
        reader = csv.reader(input, delimiter="\t") 
        for row in reader:
            metaphor_bool = int(row[0])
            word = row[1]
            if word not in metaphor_dict:
                metaphor_dict[word] = [0,0]
            if metaphor_bool == 0: #not metaphor
                metaphor_dict[word][0] += 1
            else: #is metaphor
                metaphor_dict[word][1] += 1

    """Write down any words where there are examples of metaphors, but not basic"""
    with open(counts_of_metaphors, "w") as output: 
        writer = csv.writer(output, delimiter="\t") 
        for key in metaphor_dict:
            #0 = basic, 1 = metaphor
            #find where basic count = 0, metaphor count greater than 1
            if metaphor_dict[key][0] == 0 and metaphor_dict[key][1] >= 1:
                #writer.writerow([key])
                writer.writerow((key, metaphor_dict[key][0], metaphor_dict[key][1]))

def main(): 
    #remove_columns("train.tsv", "modifiedtrain.tsv")
    count_metaphor_occurances("modifiedtrain.tsv", "counts_of_metaphors_train_w_counts.tsv")


if __name__=="__main__": 
    main() 