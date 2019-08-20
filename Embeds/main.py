from Glove import Embedding
k = Embedding(epoch=5, windows=8, embedding_dim=100)
k.fit("./data/renminribao.txt")
print(k.most_similar("美国"))
print(k.similar("美国", "日本"))
