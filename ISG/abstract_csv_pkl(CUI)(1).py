import csv
import pickle

disease = '抑郁症'
date = '(20210831)'
data = []
with open('各类疾病/'+disease+'/'+disease+'_Pubmed_Abstract_CUI编码_三类_未按照TextRank和词频进行剔除.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] != '':
            abstract = list(filter(None, row))
            data.append(abstract)

result = data
print(result[0])
table = list()
for i in range(result.__len__()):
    table.append(result[i][3:])
print(table)
with open('各类疾病/'+disease+'/CUI_abstract_'+disease+date+'.pkl', 'wb') as file:
    pickle.dump(table, file)
