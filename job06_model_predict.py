import pandas as pd
import numpy as np
from konlpy.tag import Okt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model

df = pd.read_csv("./crawling_data/naver_news_titles_20231012_104244.csv")
print(df)
df.info()

X = df["titles"]
Y = df["category"]

with open("./models/encoder.pickle", "rb") as f:
    encoder = pickle.load(f)

# 라벨 붙이는 작업
labeled_y = encoder.transform(Y) # 기존 정보를 바꾸려는 거니까 fit 없이 그냥 tranform. fit을 하면 정보를 새로 가지게 됨.
label = encoder.classes_

onehot_y = to_categorical(labeled_y)
print(onehot_y)

okt = Okt()

# 토크나이징
for i in range(len(X)):
    X[i] = okt.morphs((X[i], stem = True))
stopwords = pd.read_csv('./stopwords.csv', index_col=0)

for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)

with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_x = token.texts_to_sequences(X)

for i in range(len(tokened_x)):
    if len(tokened_x) > 21: # 문장의 길이가 21보다 크면
        tokened_x[i] = tokened_x[i][:22]
x_pad = pad_sequences(tokened_x, 21)

model = load_model('./models/') #모델 경로
preds = model.predict(x_pad)
predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predicts.append([most, second])
df['predict'] = predicts
print(df.head())

df["OX"] = 0
for i in range(len(df)):
    if df.loc[i, "category"] in df.loc[i, "predict"]:
        df.loc[i, "OX"] = "O"
    else:
        df.loc[i, "OX"] = "X"
print(df["OX"].value.counts())
print(df['OX'].value.counts()/len(df))
for i in range(len(df)):
    if df['category'][i] not in df['predict'][i]:
        print(df.iloc[i])









