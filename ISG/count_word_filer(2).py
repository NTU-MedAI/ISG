import pickle
import collections

disease = 'Sumdis'
date = '(20210904)'
data = pickle.load(open('各类疾病/'+disease+'/CUI_abstract_'+disease+date+'.pkl', 'rb'))
min_count = 5
data_words = list()
for i in range(data.__len__()):
    for j in range(data[i].__len__()):
        data_words.append(data[i][j])
word_counts = collections.Counter(data_words)
print('筛选前文本长度:', data_words.__len__())
dictionary = dict(word_counts)

# print(dictionary)
result_dict = list()
result_count = list()
keys = dictionary.keys()
print('total words count:', keys.__len__())
for key in dictionary:
    if dictionary[key] > min_count:
        result_dict.append(key)
        result_count.append(dictionary[key])
print('select words count:', result_dict.__len__())
result = dict(zip(result_dict, result_count))

select_data = list()
for i in range(data_words.__len__()):
    if data_words[i] in result:
        select_data.append(data_words[i])

result_abstract = list()
for i in range(data.__len__()):
    table = []
    for j in range(data[i].__len__()):
        if data[i][j] in result:
            table.append(data[i][j])
    result_abstract.append(table)

print('筛选后文本长度:', select_data.__len__())

with open('min_count_select_data/abstract_selected_'+disease+'CUI'+date+'.pkl', 'wb') as file:
    pickle.dump(result_abstract, file)
with open('min_count_select_data/dict_count_selected_'+disease+'CUI'+date+'.pkl', 'wb') as f:
    pickle.dump(result, f)
