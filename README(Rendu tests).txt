Pour linéaire simple avec perceptron:
- Pour 100 époques, le perceptron à une prédiction correcte à 67%
- Nous sommes donc passer à 1000 époques, et cette-fois-ci, le perceptron ne commet presque plus aucunes erreurs

Pour linéaire simple avec mlp(2,1):
- Avec un learning rate de 0.1, 1000 époques, on avait 85% de précision environ
- Avec un learning rate de 0.1, 1000 époques, on avait 96% de précision environ
- Avec un learning rate de 0.1, 1M époques, on avait plus de 99% à chaque fois

Pour linéaire multiple avec perceptron:
- Pour 1000 époques, le perceptron à une prédiction quasi aléatoire
- Nous sommes donc passer à 10000 époques, et cette-fois-ci, le perceptron ne commet presque plus aucunes erreurs

Pour linéaire multiple avec mlp(2,1):
- Avec un learning rate de 0.1, 1000 époques, on avait 98% de précision environ
- Avec un learning rate de 0.1, 10000 époques, on avait 99% de précision environ
- Avec un learning rate de 0.01, 10000 époques, on avait 30% de précision environ (Sur un point proche de la limite)


Pour XOR avec perceptron:
- De 100 à 1M d'époques, le perceptron à une prédiction nulle

Pour XOR avec mlp:
- (2,1), avec un learning rate de 0.1, 1000 époques, on avait une prédiction quasi aléatoire
- (2,1), avec un learning rate de 0.1, 10000 époques, on avait une prédiction quasi aléatoire
- (2,2,1), avec un learning rate de 0.1, peu importe le nombres d'époques, on avait une prédiction quasi aléatoire
- (2,4,1), avec un learning rate de 0.1, avec 1M d'époques, on avait une prédiction souvent supérieure à 99% (Nous pensons à de l'overfitting)

Pour cross avec mlp:
- (2,4,1), avec un learning rate de 0.1, 1000 époques, on avait une prédiction quasi aléatoire
- (2,4,1), avec un learning rate de 0.1, 10000 époques, on avait une prédiction quasi aléatoire
- (2,4,1), avec un learning rate de 0.1, 100000 époques, on avait une prédiction quasi aléatoire
- (2,5,1), avec un learning rate de 0.1, 100000 époques, on avait une prédiction quasi aléatoire
- (2,5,1), avec un learning rate de 0.1, 1M époques, on avait une prédiction de 86% (avec un point facile 0,0)
- (2,5,1), avec un learning rate de 0.1, 1M époques, on avait une prédiction de 98% (avec un point facile 1,1)
- (2,5,5,1), avec un learning rate de 0.1, 100000 époques, on avait une prédiction quasi aléatoire

Pour linéaire multiple 3 classes avec mlp:
- (3,2), avec un learning rate de 0.1, 1000 époques, erreur avec Rust (core dumped, index out of bonds)






