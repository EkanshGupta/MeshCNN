import numpy as np
import matplotlib.pyplot as plt

file1 = "meshcnn_exp3_200epochs.txt"
file2 = "meshcnn_baseline_200epochs_new.txt"
file3 = "meshcnn_exp4_200epochs.txt"
file4 = "meshcnn_exp6_200epochs.txt"
num_dict = ['1','2','3','4','5','6','7','8','9','0','.']

def preprocess_word(word, dictionary):
    str=''

    for char in word:
        if char in dictionary:
            str = str+char
    return str

def read_file_acc(file):
    with open(file,'r') as f:
        lines = f.read()

    test_acc = lines.split("TEST ACC: ")
    val=[]
    for i in range(len(test_acc)):
        if i==0:
            continue
        line = test_acc[i]
        val.append(float(preprocess_word(line.split("%")[0],num_dict)))

    return val

# print(val)
# print(len(val))
val1 = read_file_acc(file1)
val2 = read_file_acc(file2)
val3 = read_file_acc(file3)
val4 = read_file_acc(file4)

legend_list = ['Paper baseline',
                'configuration 1',
                'configuration 2',
                'configuration 3'
                ]

plt.plot(range(len(val1)),val1,'b',
         range(len(val2)),val2,'r',
         range(len(val3)),val3,'g',
         range(len(val4)),val4,'black'
         )
plt.xlabel('Epochs')
plt.ylabel('Test accuracy')
plt.title("MeshCNN accuracy for different configurations of the model parameters")
plt.legend(legend_list, loc='lower right')
plt.show()
