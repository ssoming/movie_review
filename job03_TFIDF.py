import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread
import pickle

df_reviews = pd.read_csv("datasets/reviews_2017_2022.csv")
df_reviews.info()

Tfidf = TfidfVectorizer(sublinear_tf = True)
Tfidf_matirx = Tfidf.fit_transform(df_reviews.reviews)
print(Tfidf_matirx.shape)   #83709: 형태소의 개수

with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(Tfidf, f)
mmwrite('models/Tfidf_movie_review.mtx', Tfidf_matirx)     # mmwrite: matrix 형태의 데이터 저장
