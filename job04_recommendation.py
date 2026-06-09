import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec

from job03_TFIDF import df_reviews, Tfidf_matirx


def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
    simScore = simScore[:11]
    mobieIdx = [i[0] for i in simScore]
    recmovieList = df_reviews.iloc[mobieIdx, 0]
    return recmovieList[1:11]

df_reviews = pd.read_csv('datasets/reviews_2017_2022.csv')
Tfidf_matirx = mmread('models/Tfidif_movie_review.mtx').tocsr()
with open('models/tfidf.pkl', 'rb') as f:
    Tfidf = pickle.load(f)

# movie index 사용
ref_idx = 1228

print('title', df_reviews.iloc[ref_idx, 0])
cosine_sim = linear_kernel(Tfidf_matirx[ref_idx], Tfidf_matirx)
print(cosine_sim[0])
print(len(cosine_sim))
recommendations = getRecommendation(cosine_sim)
print(recommendations)