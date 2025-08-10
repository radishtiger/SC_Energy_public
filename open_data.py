import pickle, os


filename = "set_nli_train_con_dataset.pickle"

with open(filename, 'rb') as f:
    data = pickle.load(f)


for d in data:
    print(d)