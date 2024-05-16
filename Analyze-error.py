input_file1 = "error-analysis-sophie.csv"
input_file2 = "error-analysis-lincy.csv"


"""
error-analysis-sophie.csv
zero_count: 1123
one_count: 1049

error-analysis-lincy.csv
zero_count: 1176
one_count: 956
"""


def find_predictions(input_file):
    """See if model is biased to saying metaphor or not metaphor"""
    with open (input_file,"r") as f:
        print(input_file)
        l = f.readline()
        zero_count = 0
        one_count = 0
        for line in f:
            l = line.split(",")
            true_label = l[1]
            predicted_label = l[2]
            print(predicted_label)
            if predicted_label == "0":
                zero_count += 1
            elif predicted_label == "1":
                one_count += 1
            else:
                print("error")
        print("zero_count:", zero_count)
        print("one_count:", one_count)

find_predictions(input_file1)
find_predictions(input_file2)