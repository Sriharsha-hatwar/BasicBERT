input_file = "error-analysis-sophie.csv"
"""See if model is biased to saying metaphor or not metaphor"""
with open (input_file,"r") as f:
    l = f.readline()
    zero_count = 0
    one_count = 0
    for line in f:
        l = line.split(",")
        true_label = l[1]
        predicted_label = l[2]
        if predicted_label == "0":
            zero_count += 1
        elif predicted_label == "1":
            one_count += 1
        else:
            print("error")
    print("zero_count:", zero_count)
    print("one_count:", one_count)

        