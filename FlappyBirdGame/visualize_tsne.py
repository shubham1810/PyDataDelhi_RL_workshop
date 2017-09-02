import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data = np.load('./vector_data_frame.npy')
X = data[:, :-1]
y = data[:, -1]

print y.shape

tsne = TSNE(n_components=2)

X_emb = tsne.fit_transform(X)

plt.plot(X_emb[:, 0], X_emb[:, 1], 'o-')#, c=range(X.shape[0]), s=75)

plt.show()