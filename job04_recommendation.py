import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
import pickle
from konlpy.tag import Okt
from gensim.models import Word2Vec


def getRecommendation(cosine_sim):
    simScore = list(enumerate(cosine_sim[-1]))
    simScore = sorted(simScore, key=lambda x: x[1], reverse=True)
    simScore = simScore[:11]
    mobieIdx = [i[0] for i in simScore]
    recmovieList = df_reviews.iloc[mobieIdx, 0]
#    return recmovieList[1:11]
    return recmovieList[0:11]


df_reviews = pd.read_csv('datasets/reviews_2017_2022.csv')
Tfidf_matirx = mmread('models/Tfidf_movie_review.mtx').tocsr()
with open('models/tfidf.pkl', 'rb') as f:
    Tfidf = pickle.load(f)

# # movie index 사용
# ref_idx = 1228
#
# print('title', df_reviews.iloc[ref_idx, 0])
# cosine_sim = linear_kernel(Tfidf_matirx[ref_idx], Tfidf_matirx)
# print(cosine_sim[0])
# print(len(cosine_sim))
# recommendations = getRecommendation(cosine_sim)
# print(recommendations[1:11])


# Keyword 사용
embedding_model = Word2Vec.load('models/word2vec_movie_review.model')
keyword = '재앙'
if keyword not in list(embedding_model.wv.index_to_key):
    print('모르는 단어입니다.')
else:
    sim_word = embedding_model.wv.most_similar(keyword, topn = 10)
    print(sim_word)

    sentence = [keyword] * 11
    count = 10
    for word, _ in sim_word:        # 받아서 사용 안 할 변수는 '_'로
        sentence = sentence + [word] * count
        count = count - 1
    # 유사도가 높을수록 단어 수 증가
    print(sentence)
    sentence = ' '.join(sentence)
    print(sentence)

    sentence_vec = Tfidf.transform([sentence])
    cosine_sim = linear_kernel(sentence_vec, Tfidf_matirx)
    recommendation = getRecommendation(cosine_sim)
    print(recommendation)