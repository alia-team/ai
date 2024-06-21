from mlp import MLP
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["figure.figsize"] = (10, 10)

loss = False

# Linear Simple
# MLP (2, 1)   : OK
npl = (2, 1)
mlp = MLP(npl)

training_dataset = np.array([
      [1, 1],
      [2, 3],
      [3, 3]
])
labels = np.array([
    [1.0],
    [-1.0],
    [-1.0]
])
res = mlp.train(training_dataset, labels, training_dataset, labels, 0.1, 100000, True)
print(res[len(res)-1][0])
plt.scatter([p[0] for p in training_dataset], [p[1] for p in training_dataset], c=[l[0] for l in labels])
x = np.linspace(0, 4, 300)
y = np.linspace(0, 4, 300)
X, Y = np.meshgrid(x, y)
Z = np.array([[1 if mlp.predict([x, y],True)[0]>0 else -1 for x in x] for y in y])
plt.contourf(X, Y, Z, alpha=0.5)
plt.title('Linear Simple')
plt.show()
plt.clf()

if loss:
    plt.plot([i*100 for i in range(len(res))], [r[0] for r in res], color='blue')
    plt.plot([i*100 for i in range(len(res))], [r[1] for r in res], color='red')
    plt.legend(['train dataset', 'test dataset'])
    plt.title('Linear Simple')
    plt.show()
    plt.clf()

# Linear Multiple :
# MLP (2, 1): OK
npl = (2, 1)
mlp = MLP(npl)

training_dataset = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])])
labels = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])
print(max(training_dataset[:, 0]), max(training_dataset[:, 1]))
res = mlp.train(training_dataset, labels, training_dataset, labels, 0.1, 10000, True)
print(res[len(res)-1][0])
plt.scatter([p[0] for p in training_dataset], [p[1] for p in training_dataset], c=[l[0] for l in labels])
x = np.linspace(0, 4, 300)
y = np.linspace(0, 4, 300)
X, Y = np.meshgrid(x, y)
Z = np.array([[1 if mlp.predict([x, y],True)[0]>0 else -1 for x in x] for y in y])
plt.contourf(X, Y, Z, alpha=0.5)
plt.title('Linear Multiple')
plt.show()
plt.clf()

if loss:
    plt.plot([i*100 for i in range(len(res))], [r[0] for r in res], color='blue')
    plt.plot([i*100 for i in range(len(res))], [r[1] for r in res], color='red')
    plt.legend(['train dataset', 'test dataset'])
    plt.title('Linear Multiple')
    plt.show()
    plt.clf()

# XOR
# MLP (2, 2, 1): 
npl = (2, 4, 1)
mlp = MLP(npl)
training_dataset = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
labels = np.array([[1], [1], [-1], [-1]])

res = mlp.train(training_dataset, labels, training_dataset, labels, 0.1, 1000000, True)
print(res[len(res)-1][0])
plt.scatter([p[0] for p in training_dataset], [p[1] for p in training_dataset], c=[l[0] for l in labels])
# add background color based on the prediction
x = np.linspace(0, 1, 300)
y = np.linspace(0, 1, 300)
X, Y = np.meshgrid(x, y)
Z = np.array([[1 if mlp.predict([x, y],True)[0]>0 else -1 for x in x] for y in y])
plt.contourf(X, Y, Z, alpha=0.5)
plt.title('XOR')
plt.show()
plt.clf()

if loss:
    plt.plot([i*100 for i in range(len(res))], [r[0] for r in res], color='blue')
    plt.plot([i*100 for i in range(len(res))], [r[1] for r in res], color='red')
    plt.legend(['train dataset', 'test dataset'])
    plt.title('XOR')
    plt.show()
    plt.clf()


# Cross
# MLP (2, 4, 1)
npl = (2, 4, 1)
mlp = MLP(npl)

# Initialize the training dataset and labels
training_dataset = np.random.random((500, 2)) * 2.0 - 1.0
labels = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in training_dataset])
labels = labels.reshape(-1, 1)  # Ensure labels are in the correct shape

res = mlp.train(training_dataset, labels, training_dataset, labels, 0.1, 100000, True)
print(res[len(res)-1][0])
plt.scatter([p[0] for p in training_dataset], [p[1] for p in training_dataset], c=[l[0] for l in labels])
# add background color based on the prediction
x = np.linspace(-1, 1, 300)
y = np.linspace(-1, 1, 300)
X, Y = np.meshgrid(x, y)
Z = np.array([[1 if mlp.predict([x, y],True)[0]>0 else -1 for x in x] for y in y])
plt.contourf(X, Y, Z, alpha=0.5)
plt.title('Cross')
plt.show()
plt.clf()

if loss:
    plt.plot([i*100 for i in range(len(res))], [r[0] for r in res], color='blue')
    plt.plot([i*100 for i in range(len(res))], [r[1] for r in res], color='red')
    plt.legend(['train dataset', 'test dataset'])
    plt.title('Cross')
    plt.show()
    plt.clf()


# Multi linear 3 classes
# MLP (2, 3)
npl = (2, 3,3)
mlp = MLP(npl)

# Initialize the training dataset and labels
training_dataset = np.random.random((500, 2)) * 2.0 - 1.0
labels = np.array([[1, -1, -1] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
              [-1, 1, -1] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
              [-1, -1, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
              [-1, -1, -1]for p in training_dataset])

training_dataset = training_dataset[[not np.all(arr == [0, 0, 0]) for arr in labels]]
labels = labels[[not np.all(arr == [0, 0, 0]) for arr in labels]]

res = mlp.train(training_dataset, labels, training_dataset, labels, 0.1, 1000, True)
print(res[len(res)-1][0])
plt.scatter([p[0] for p in training_dataset], [p[1] for p in training_dataset], c=[l[0] for l in labels])
if loss:
    plt.plot([i*100 for i in range(len(res))], [r[0] for r in res], color='blue')
    plt.plot([i*100 for i in range(len(res))], [r[1] for r in res], color='red')
    plt.legend(['train dataset', 'test dataset'])
    plt.title('Multi linear 3 classes')
    plt.show()
    plt.clf()
