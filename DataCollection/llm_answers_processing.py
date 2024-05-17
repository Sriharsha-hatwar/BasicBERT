import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd

paths = ['chat_gpt_ans_1.txt', 'chat_gpt_ans_2.txt', 'chat_gpt_ans_3.txt']
answers = []
for p in paths:
    with open(p, 'r') as f:
        lines = f.readlines()
        for l in lines:
            answers.append(l.strip())

records = []
sent_id = 1
#temp = answers[0:2]
for ans in answers:
    sent = ans.split(':')[1]
    sent_tokens = sent.split()
    fgpos_tags = pos_tag(sent_tokens)
    pos_tags = pos_tag(sent_tokens, tagset='universal')
    
    assert len(pos_tags) == len(fgpos_tags)
    assert len(fgpos_tags) == len(sent_tokens)
    
    for i in range(len(fgpos_tags)):
        fgp_word, fgp_tag = fgpos_tags[i]
        p_word, p_tag = pos_tags[i]
        if fgp_tag.isalnum() and p_word == fgp_word:    
            records.append({'index': 'gpt_id_'+str(sent_id), 'label':0, 'sentence':sent, 'POS':p_tag , 'FGPOS':fgp_tag , 'w_index':i })
        
        i += 1
            
    
    sent_id += 1

out_path = 'gpt_records.tsv'
df = pd.DataFrame(records)
df.to_csv(out_path, sep='\t', index=False)

