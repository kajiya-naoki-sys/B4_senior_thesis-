import os
import csv
import matplotlib.pyplot as plt      

path = os.getcwd()
file_path = path + "/result.csv"

with open(file_path) as f:
    csvreader = csv.reader(f)
    result_data = [row for row in csvreader]
    del result_data[0]
    del result_data[-1]

    '''
    print('---')
    print(result_data)
    print('---')   
    '''
    
    for row in result_data:
        del row[0]
'''
print('---')
print(result_data)
print('---')
'''

def sep(l, data, num):
    for row in data:
        l.append(float(row[num])*100.0)
    return l  
accuracy = []
precision =[]
recall = []
f1 = []
accuracy = sep(accuracy, result_data, 0)
precision = sep(precision, result_data, 1)
recall = sep(recall, result_data, 2)
f1 = sep(f1, result_data, 3)

'''
print('---')
print(accuracy)
print(precision)
print(recall)
print(f1)
print('---')
'''

fig, ax = plt.subplots()
plt.boxplot((accuracy, precision, recall, f1), labels=['accuracy', 'precision', 'recall', 'f1'], whis=[0, 100], showmeans=True, meanprops={"marker":"x"})
plt.xlabel('metrics')
plt.ylabel('score(%)')
plt.show()