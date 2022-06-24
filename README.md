# mnist_adv_exp_generator
Generating adversarial examples for a convolutional neural network by the optimization algorithm differential evolution.
According to Ian Goodfellow deep learning networks are vulnerable to adversarial examples: handcrafted pictures which are wrongly predicted by a classifier.
I have used the evolutive algorithm differential evolution to choose the best pertubations in order to minimize the confidence level towards the correct class. Adversarial examples generated against my net should remain adversarial also for other networks.
