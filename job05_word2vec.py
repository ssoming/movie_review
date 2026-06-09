import pandas as pd
from gensim.models import word2vec, Word2Vec

df_reviews = pd.read_csv('datasets/reviews_2017_2022.csv')
df_reviews.info()

reviews = list(df_reviews.reviews)
print(reviews[0])

tokens = []
for sentence in reviews:
    token = sentence.split()
    tokens.append(token)
print(tokens[0])

embedding_model = Word2Vec(tokens, vector_size=100, window=4,
                           min_count=20, workers=4, epochs=100, sg=1)
# vector_size: 차원 축소
# min_count: n번 이상 나올 때 의미 벡터로 사용.
embedding_model.save('models/word2vec_movie_review.model')
print(list(embedding_model.wv.index_to_key))
print(len(embedding_model.wv.index_to_key))
