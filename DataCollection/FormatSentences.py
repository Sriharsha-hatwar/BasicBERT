import nltk
import os
import operator
import ssl

"""
Format found Merrium-Webster sentences for use in BasicBert
"""

def formatSentences(input, output):
    """ Saves metaphor bool, sentence, w_index in standard tsv """
    replace_chars = ["{wi}", "{/wi}", "{it}", "{/it}", "{phrase}", "{/phrase}"]
    remove_chars = ["{ldquo}", "{/ldquo}", "{rdquo}", "{/rdquo}", "\n", "{gloss}=", "{\gloss}", "{d_link|"]
    format_pos = {"verb":"VERB", "adjective":"ADJ", "preposition":"ADP", "noun":"NOUN", "adjective suffix":"ADJ", "adverb":"ADV", "phrasal verb":"VERB"}
    with open(input, "r") as i:
        with open(output, "w+") as o:
            for line in i:
                for c in replace_chars:
                    line = line.replace(c, "*")
                for c in remove_chars:
                    line = line.replace(c, "")
                pos_split = line.split("\t")
                split_sentence = pos_split[0].split("*")

                try:
                    word = split_sentence[1]
                    count = split_sentence[0]
                    pos_1 = pos_split[1]
                    pos_1 = format_pos[pos_1]
                    pos_2 = nltk.pos_tag([word])[0][1]
                    index = operator.countOf(count, " ")
                    #print("0", line.replace("*", ""), str(index))
                    sentence = pos_split[0].replace("*", "")
                    write_line = "MW" + "\t"+  "0" + "\t" + sentence + "\t" + pos_1 + "\t"+ pos_2 + "\t" + str(index) + "\n"
                    o.write(write_line)
                except:
                    print("error")
                    print(line)
                
    return
 
def formatBasicSentences():
    input_file = "VUA20-BasicSentences/BasicSentences_0-1607.txt"
    output_file = "VUA20-BasicSentences/FormattedBasicSentences_0-1607.tsv"
    with open (output_file, "w+"):
        pass

    if os.path.exists(input_file) == False:
        print("No input file of that name")
        quit()

    formatSentences(input_file, output_file)
    
def main():
    #getBasicSentences2("","")
    formatBasicSentences()

    return
    
if __name__=="__main__": 
    main() 
