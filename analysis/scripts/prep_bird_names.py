import json
from random import shuffle

with open("../../networks/birds_levit/idx2classes.json") as f:
    id2label = json.load(f)

for k in id2label:
    id2label[k] = id2label[k].upper().replace(' ','_').replace('-','_').replace('\'','')

print(type(id2label.values()))

in_set = []
not_in_set = []

with (open('../bird_names/raw.txt','r') as f):
    for line in f:
        if line.strip():
            line = line.replace('\n','').upper().replace(' ','_').replace('-','_').replace('\'','')
            if line in id2label.values():
                in_set.append(line)
            else:
                not_in_set.append(line)

shuffle(not_in_set)
n = len(not_in_set)
print(n)
selected = not_in_set[:int(n*0.005)]
print(selected)

with (open('../bird_names/selected.txt','w') as w):
    for e in selected:
        w.write(f'{e}\n')