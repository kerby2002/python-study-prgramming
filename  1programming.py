import numpy as np
import matplotlib.pyplot as plt
weights = 10 * np.random.random(2)
data = 10 * np.random.random((100,2))
y = np.sum(data,axis=1)
x = np.linspace(-50, 50, 100)
y_= np.linspace(-50, 50, 100)
X, Y = np.meshgrid(x, y)

def loss(w):
    yhat = data @ w
    return np.mean((yhat - y)**2)

@np.vectorize
def Loss(x, y):
    w = np.array([x, y])
    yhat = data @ w
    return np.mean((yhat - y)**2)

Z = Loss(X, Y)

def dloss(w):
    yhat = data @ w
    return np.array([2 * np.mean((yhat - y) * data[:, 0]), 2 * np.mean((yhat - y) * data[:, 1])])

epochs = 1000
losses = []
path = []
learning_rate = 0.01
for i in range(epochs):
    losses.append(loss(weights))
    path.append((weights[0], weights[1], loss(weights)))
    d = dloss(weights)
    weights = weights - learning_rate * d 

evals=np.arange(epochs)
plt.plot(evals, losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = zip(*path)
ax.plot(x, y, z)
ax.plot_surface(X, Y, Z, alpha=0.5)

plt.show()
x = np.array([20,57])
print(weights @ x)