import pickle


disease = 'Sumdis'
data = pickle.load(open('tokens/tokens_' + disease + '(20210904).pkl', 'rb'))
print(data.__len__())


text8_table = list()
for i in range(data.__len__()):
    for j in range(data[i].__len__()):
        text8_table.append((data[i][j]))


with open('各类疾病/'+disease+'/text8_tokens_'+disease+'(20210904).pkl', 'wb') as file:
    pickle.dump(text8_table, file)

