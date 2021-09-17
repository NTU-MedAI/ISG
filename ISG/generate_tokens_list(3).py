import pickle
import tensorflow as tf

disease = 'Sumdis'
date = '(20210904)'
file = open('min_count_select_data/abstract_selected_'+disease+'CUI'+date+'.pkl', 'rb')
data = pickle.load(file)
file.close()
# print(data)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, lower=False)
tokenizer.fit_on_texts(data)
keys_list = tokenizer.word_index
print(tokenizer.word_index)
tokens = tokenizer.texts_to_sequences(data)
print(tokenizer.texts_to_sequences(data))
with open('tokens_list/tokens_list—'+disease+date+'.pkl', 'wb') as file:
    pickle.dump(keys_list, file)
with open('tokens/tokens_'+disease+date+'.pkl', 'wb') as f:
    pickle.dump(tokens, f)