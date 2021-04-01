import csv

d = {}

with open('archive/dataset_TSMC2014_TKY.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        # print(row)
        if row[0] not in d:
            d[row[0]] = []
        f = row[-1]
        d[row[0]].append(row)
for each in d:
    print(len(d[each]))