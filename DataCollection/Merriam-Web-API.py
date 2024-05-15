
import sys
import requests
import json
"""
Make calls to Merriam-Webster API given list of target words
and records JSON results to file
Requires MW API Key as argument
"""

def parse_json_obj():
    with open('data.json', 'r') as i:
        with open('MW-sentences', 'a') as o:
            jsonObj = json.load(i)
            define_word = jsonObj[0]["def"][0]["sseq"]
            for x in define_word:
                try:
                    if len(x[0][1]['dt']) > 1:
                        sentence = x[0][1]['dt'][1][1][0]['t'].replace("{wi}", "").replace("{/wi}", "")
                        print(sentence)
                    o.write(sentence+ "\n")
                except:
                    continue

def combine_json_obj(input_file, list_of_words, key):
    """Read json obj from file, add new json objects retrieved from MW API
    with all words from list_of_words
    key: API key
    """
    with open(input_file, 'r') as i:
        json_words = json.load(i)

    for word in list_of_words:
        r = get_literal_sentence(word, key)
        if r != None:
            json_words.append(r)
    with open(input_file, "w") as i:
        json.dump(json_words, i, indent = 4)

def get_literal_sentence(word, key):
    """Make call to Merriam-Webster API to retrieve json obj for "word"
    input
        key: MW API key
    """
    #Max 1000 calls per day
    #https://www.dictionaryapi.com/api/v3/references/collegiate/json/voluminous?key=your-api-key
    url = "https://www.dictionaryapi.com/api/v3/references/collegiate/json/" + word + "?key=" + key
    try:
        r = requests.get(url)
        r = r.json()
        #jsonObj = r.load()
        return r
    except:
        return

def get_words(input_file, min, max):
    """Read list of words from file into array
    input:
        input_file: file to read from
        min, max: line numbers to save words from
    """
    word_list = []
    with open(input_file, 'r') as i:
        for x in range(0,min):
            line = i.readline()
        for x in range(min,max):
            line = i.readline().strip("\n")
            word_list.append(line)
    print(word_list)
    return word_list

import os
def main():
    """Pass API key as argument"""
    arguments = sys.argv
    key = arguments[1]

    #min = where to start calling on target words
    min = 0
    max = 300

    word_list = get_words("words_w_no_basic_def.tsv", min, max)
    data_file_name = "data_" + str(min+1) + "-" + str(max) + ".json" 
    print(data_file_name)
    
    if os.path.exists(data_file_name):
        print("File already exists")
        quit()
    else:
        print("Creating File")
        with open(data_file_name, "w+") as f:
            f.write("[]")
        combine_json_obj(data_file_name, word_list, key)
    
if __name__=="__main__": 
    main() 
