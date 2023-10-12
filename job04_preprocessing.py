import pandas as pd
import numpy as np
from konlpy.tag import Okt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

pd.set_option("display.unicode.east_asian_width", True)
df = pd.read_csv("./crawling_data/naver_news_titles_20231012_162645.csv")
df.info()

X = df["titles"]
Y = df["category"]

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
label = encoder.classes_

with open("./models/encoder.pickle", "wb") as f:
    pickle.dump(encoder, f)

onehot_y = to_categorical(labeled_y)

okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i])

stopwords = pd.read_csv("./stopwords.csv", index_col=0)
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1:
            if X[j][i] not in list(stopwords["stopword"]):
                words.append(X[j][i])
    X[j] = " ".join(words)

token = Tokenizer()  # 형태소들을 숫자로 변환해주는 것을 토크나이저라고 함
token.fit_on_texts(X)  # 형태소에 라벨을 부여
tokened_x = token.texts_to_sequences(X)  # 라벨로 된 리스트를 만들어줌
wordsize = len(token.word_index) + 1  # 라벨링 할 때는 0을 쓰지 않으므로 +1

with open("./models/news_token.pickle", "wb") as f:
    pickle.dump(token, f)

max_len = max(len(s) for s in tokened_x)
x_pad = pad_sequences(tokened_x, maxlen=max_len)
print(x_pad[:3])

X_train, X_test, Y_train, Y_test = train_test_split(x_pad, onehot_y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
# 기존 코드
# np.save("./crawling_data/news_data_max_{}_wordsize".format(max, wordsize), xy)

# 수정된 코드
np.save(f"./crawling_data/news_data_max_{max_len}_wordsize_{wordsize}", xy)
