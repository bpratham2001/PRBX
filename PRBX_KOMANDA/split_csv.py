import os
import csv
#abs_path = '/shared/storage/cs/studentscratch/pb1028/new_venv/PRBX_komanda/interpolated.csv'
abs_path = os.path.join('/shared/storage/cs/studentscratch/pb1028/new_venv/PRBX_komanda/', 'interpolated.csv')
path = 'interpolated.csv'
train = 9000
test = 3000
def split(output, start=0, end=1):
    with open(path, 'r', newline='') as f,open(output, 'w', newline='') as f_out:
        reader = csv.reader(f, delimiter=",")
        writer = csv.writer(f_out, delimiter=",")
        for counter,row in enumerate(reader):
                if counter >= end:
                    break
                if counter >= start:
                    writer.writerow(row)  
    with open(output, 'r') as f:
        print(str(len(list(csv.reader(f,delimiter = ",")))) + " data points in " + output) 
print("starting split_csv ...")
with open(path, 'r') as f:
    reader = csv.reader(f,delimiter = ",")
    data = list(reader)
    n = len(data)
    print(str(n) + " data points in interpolated.csv")    
split("interpolated_train.csv", end=train)
split("interpolated_test.csv", start=train, end=(train+test))