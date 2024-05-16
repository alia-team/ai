from mlp import MLP
import matplotlib.pyplot as plt
import numpy as np

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

mlp.train(training_dataset, labels, 0.1, 1000000, True)

for k in range(len(training_dataset)):
    output = mlp.predict(training_dataset[k], True)
    print(output)


# Linear Simple
# MLP (2, 1): OK
plt.scatter(training_dataset[0, 0], training_dataset[0, 1], color='blue')
plt.scatter(training_dataset[1:3,0], training_dataset[1:3,1], color='red')
plt.show()
plt.clf()


# Linear Multiple :
# MLP (2, 1): OK
training_dataset = np.concatenate([np.random.random((50,2)) * 0.9 + np.array([1, 1]), np.random.random((50,2)) * 0.9 + np.array([2, 2])])
labels = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

plt.scatter(training_dataset[0:50, 0], training_dataset[0:50, 1], color='blue')
plt.scatter(training_dataset[50:100,0], training_dataset[50:100,1], color='red')
plt.show()
plt.clf()


# XOR
# MLP (2, 2, 1): 
training_dataset = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
labels = np.array([1, 1, -1, -1])

plt.scatter(training_dataset[0:2, 0], training_dataset[0:2, 1], color='blue')
plt.scatter(training_dataset[2:4,0], training_dataset[2:4,1], color='red')
plt.show()
plt.clf()


# Cross
# MLP (2, 4, 1): 
training_dataset = np.random.random((500, 2)) * 2.0 - 1.0
labels = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in training_dataset])

plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(training_dataset)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(training_dataset)))))[:,1], color='blue')
plt.scatter(np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(training_dataset)))))[:,0], np.array(list(map(lambda elt : elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(training_dataset)))))[:,1], color='red')
plt.show()
plt.clf()
