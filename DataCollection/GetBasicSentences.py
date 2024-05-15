
import sys
import json
import os

"""
Parse JSON objects received from Merrium-Webster for sentences + Part of Speech
"""


def parse_json_obj(json_file, BasicSentences):
    with open(json_file, 'r') as i:
        with open(BasicSentences, 'a') as o:
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

def load_words(data_file_name):
    """Loads json file into json object"""
    with open( data_file_name, "r") as f:
        l = json.load(f)
    return l

def write_to_file(d, s, pos):
    d.write(s)
    d.write("\t")
    d.write(pos)
    d.write("\n")
    return

def getBasicSentences(json_sentences, data_file_name):
    not_found = set()
    """Read through json objects for example basic sentences"""
    with open (data_file_name, "a") as d:
        for w in json_sentences:
            found_word = False
            try:
                pos = w[0]['fl']
                s = w[0]['def'][0]['sseq'][0][2][1]['dt'][1][1][1]['t']
                write_to_file(d, s, pos)
                found_word = True
            except:
                pass
            try:
                pos = w[0]['fl']
                s = w[0]['def'][0]['sseq'][0][2][1]['dt'][1][1][1]['t']
                write_to_file(d, s, pos)
                found_word = True
            except:
                pass
            try:
                pos = w[0]['fl']
                s = w[0]['def'][0]['sseq'][0][1][1]['dt'][1][1][1]['t']
                write_to_file(d, s, pos)
                found_word = True
            except:
                pass
            try:
                pos = w[0]['fl']
                s = w[0]['def'][0]['sseq'][0][1][1]['dt'][1][1][2]['t']
                write_to_file(d, s, pos)
                found_word = True
            except:
                pass
            try:
                pos = w[0]['fl']
                s = w[0]["def"][0]["sseq"][0][0][1]["dt"][1][1][0]['t']
                write_to_file(d, s, pos)
                found_word = True
            except:
                pass
            try:
                pos = w[0]['fl']
                s = w[0]["def"][0]["sseq"][1][0][1]["dt"][1][1][0]['t']
                write_to_file(d, s, pos)
                found_word = True
            except:
                pass
            try:
                pos = w[0]['fl']
                s = w[0]['def'][0]['sseq'][0][0][1]['sdsense']['dt'][1][1][0]['t']
                write_to_file(d, s, pos)
                found_word = True
            except:
                pass
            try:
                s = w[0]['def'][0]['sseq'][0][0][1]['dt'][1][1][2]['t']
                found_word = True
            except:
                pass
            try:
                pos = w[0]['fl']
                s = w[0]['def'][0]['sseq'][0][0][1]['dt'][1][1][1]['t']
                write_to_file(d, s, pos)
                found_word = True
            except:
                pass
            if found_word == False:
                try:
                    print(w[0]["meta"]["id"].split(":")[0])
                    not_found.add(w[0]["meta"]["id"].split(":")[0])
                except:
                    print(w)
                    print("\n")
    with open("not_found.txt", "a") as n:
        for f in not_found:
            n.write(f)
            n.write("\n")

def check_for_sentences(input_file):
    json_sentences = load_words(input_file)
    i = 0
    for w in json_sentences:
        try:
            print(w[0]['fl'])
            
        except:
            print("Not found")
            pass

def getAllBasicSentences():
    """
    Loops through all JSON files w/ merriam webster results
    Keeps sentences with literal uses of words
    """
    input_files = ["JSON_obj_MW/data_0-300.json", "JSON_obj_MW/data_301-700.json", "JSON_obj_MW/data_701-1000.json", "JSON_obj_MW/data_1001-1607.json"]
    output_name = "BasicSentences_0-1607.txt"
    with open (output_name, "w+"):
        pass
    with open ("not_found.txt", "w+"):
        pass

    for i in input_files:
        print(i)
        if os.path.exists(i) == False:
            print("No input file of that name")
            quit()
        w = load_words(i)
        getBasicSentences(w, output_name)
    

def main():
    input_files = ["JSON_obj_MW/data_0-300.json", "JSON_obj_MW/data_301-700.json", "JSON_obj_MW/data_701-1000.json", "JSON_obj_MW/data_1001-1607.json"]
    getAllBasicSentences()
    #check_for_sentences(input_files[0])
    return
    
if __name__=="__main__": 
    main() 
