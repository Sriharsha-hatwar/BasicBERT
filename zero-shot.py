import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from vllm import LLM, SamplingParams
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def return_scores(predicted_labels, true_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return f1, precision, recall, accuracy

def check_convert(predictions, gold_label):
    final_predictions = []
    final_gold_labels = []
    count = 0
    for index, i in enumerate(predictions):
        try : 
            final_i = int(i)
            final_predictions.append(final_i)
            final_gold_labels.append(int(gold_label[index]))
        except:
            count+=1
            pass
    print("Number of mismatches : ", count)
    return final_predictions, final_gold_labels

def predict_metaphors(tsv_file_path, model):
    # Prepare the input for the model
    df = pd.read_csv(tsv_file_path, sep='\t')
    prompt = """
    You are the world's best metaphor detection service. Use both Metaphor Identification procedure and Selectional Preference Violation to determine if a given sentence is a metaphor or not.  

Metaphor identification procedure is a procedure that given a focused lexical unit  and a sentence having that lexical unit, it determines if the basic meaning (with just the lexical unit) and the contextual meaning (used in the sentence) varies. If it Varies, it means its a metaphor, else it is not a metaphor

Selectional preference Violation is a procedure that given a focused lexical unit and a sentence having that lexical unit determines if the incogruity between the lexical unit and the sourrounding words of a given sentence is a lot. If it is varies then it means that it a metaphor, else it is not a metaphor. 

For example :  
Lexical unit :  "head" , sentence : The 63-year-old head of Pembridge Investments , through which the bid is being mounted says , ‘ rule number one in this business is : the more luxurious the luncheon rooms at headquarters , the more inefficient the business label : '1' 

Lexical unit :  "Pembridge" , sentence : The 63-year-old head of Pembridge Investments , through which the bid is being mounted says , ‘ rule number one in this business is : the more luxurious the luncheon rooms at headquarters , the more inefficient the business label : '0' 

Now given this information about the techniques and examples determine if a sentence is a metaphor or not and provide '1' if its a metaphor and '0' if its not a metaphor , dont output anything except '1' or '0'.  

Lexical unit : '{}', sentence : {}. label : 
    """

    # create the huge list with prompt applied to it. 
    prompt_list = []
    gold_labels = df["label"].tolist()

    for index, row in df.iterrows():
        word = row["sentence"].split()[row["w_index"]]
        filled_prompt = prompt.format(word, row["sentence"])
        prompt_list.append(filled_prompt)
    print("len of prompt_list : ", len(prompt_list))
    
    # Get model predictions
    sampling_params = SamplingParams(max_tokens=1, temperature=0.05, top_p = 0.95, skip_special_tokens=True)
    print("Inference using Mistral 7b V0.2")
    with torch.no_grad():
        output = model.generate(prompt_list, sampling_params)
        interim_predictions = [output[i].outputs[0].text.strip() for i in range(len(output))]
    #print(interim_predictions)
    # Check if the list has either 0 or 1 
    int_predictions, final_gold_labels = check_convert(interim_predictions, gold_labels)
    #df["predicted_label"] = int_predictions

    # Now calculate the accuracy, f1, recall, precision.
    f1, pre, rec, acc = return_scores(int_predictions, final_gold_labels)
    print("Accuracy score : ", acc)
    print("F1 score : ", f1)
    print("Precision score : ", pre)
    print("Recall score : ", rec)

def get_model(model_name):
    model = LLM(model=model_name,
                    tokenizer=model_name, 
                    tensor_parallel_size=torch.cuda.device_count(), 
                    seed=24, 
                    dtype=torch.float16,
                    enforce_eager=True,
                    max_num_seqs=128, 
                    download_dir='/work/pi_dhruveshpate_umass_edu/shatwar/.cache/hub'
                )
    return model

if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model = get_model(model_name)
    input_tsv = 'data/VUA20/test.tsv'  # Path to your input TSV file
    output_tsv = 'data/VUA20/zero-shot-test.tsv'  # Path to the output TSV file
    predict_metaphors(input_tsv, model)
