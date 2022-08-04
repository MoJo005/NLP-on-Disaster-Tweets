from transformers import TFBertModel, AutoTokenizer
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import Model
import streamlit as st
import re
import numpy as np
import tensorflow as tf




#============== Model Architecture =================
def model_architecture(bert_model, max_len):
    # Model architecture

    weight_initializer = GlorotNormal(seed=42)

    # DistilBERT Layers
    input_ids_layer = Input(shape=(max_len,), name='input_ids', dtype='int32')
    input_attention_layer = Input(shape=(max_len,), name='input_attention', dtype='int32')
    last_hidden_state = bert_model(input_ids_layer,input_attention_layer)[1]
#     cls_token = last_hidden_state[:, 0, :]

    # NN layer
    X = Dropout(0.2)(last_hidden_state)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.3)(X)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.3)(X)
    X = Dense(16, activation='relu')(X)
    
    output = Dense(1,activation='sigmoid',kernel_initializer=weight_initializer)(X)

    model_bert = Model(inputs=[input_ids_layer, input_attention_layer], outputs=output)
    
    return model_bert




#=========== Importing Model and Tokenizer=========
@st.cache(allow_output_mutation=True)
def model_params(max_len):
    bert_model=TFBertModel.from_pretrained('Model Parameters/bert-model')
    # bert_tokenizer=AutoTokenizer.from_pretrained('Model Parameters/bert-tokenizer')
    model_bert=model_architecture(bert_model, max_len)
    model_bert.load_weights('Model Parameters/bert_weights.h5')
    
    return model_bert



#================Text Preprocessing============
# 1 Converting to lower-case
def lowercase(text):
    return text.lower()

# 2 Decontracting the text
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"\'cause", " because", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)
    return phrase

# 3 Removing URLs
def remove_url(text):
    text= re.sub(r'https?://\S+|www\.\S+', '', text)
    return text

# 4 Removing HTMLs
def remove_html(text):
    text=re.sub(r'<.*?>','',text)
    return text


def basic_preprocessing(text):
    
    text=lowercase(text)           # 1
    text=decontracted(text)        # 2    
    text=remove_url(text)          # 3
    text=remove_html(text)         # 4
    # text=remove_slangs(text)       # 5
    
    return text


# ========= Text Encoding =================
def text_encoding(tokenizer, texts, max_length):
    
    """This function return the text embeddings after tokenization and padding the text."""
    batch_size=256
    input_ids = []
    attention_mask = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length, padding='max_length',
                                             truncation=True, return_attention_mask=True,
                                             return_token_type_ids=False)
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])
    
    
    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)









def main_function(text):

    max_len=64
    model_bert = model_params(max_len)

    # 1 Preprocessing the text
    text=basic_preprocessing(text)
    text_list=[]
    text_list.append(text)

    # 2 Encoding the text data
    bert_tokenizer=AutoTokenizer.from_pretrained('Model Parameters/bert-tokenizer')
    text_ids, text_attention=text_encoding(bert_tokenizer, text_list, max_len)

    # Modeling 
    pred_proba=model_bert.predict([text_ids, text_attention])
    prediction= np.where(pred_proba>0.5, 1, 0)

    return prediction




