import pickle

count = 40
date = '(20210904)'
disease = 'Sumdis'
data = pickle.load(open('tokens/tokens_'+disease+date+'.pkl', 'rb'))
print(data.__len__())

conformity_abstract = list()
table = list()
text8_table = list()
for i in range(data.__len__()):
    for j in range(data[i].__len__()):
        table.append(data[i][j])
        # text8_table.append((data[i][j]))

    if i % count == 0:
        conformity_abstract.append(table)
        table = []
print(conformity_abstract.__len__())

with open('tokens/conformity_' + str(count) + '_tokens_'+disease+date+'.pkl', 'wb') as file:
    pickle.dump(conformity_abstract, file)

# with open('8_cancer_data/text8_tokens/text8_tokens_乳腺癌(20210630).pkl', 'wb') as file:
#     pickle.dump(text8_table, file)
