# information-theory-DL
Term paper by Kirill Tyshchuk

Examining neural networks with modern methods of mutual information estimation.

* Implementing MINE (mine.py)

* Testing the implementation (mixture.py, mine_test_mixture.py, mine_test_from_paper.py, mine_test_infinity.py, mine_test_tishby_ds.py)
* Get embeddings from ResNets, Inception and Xception networks on cats vs dogs dataset and train classifiers on them. (images_to_embeddings.py, train_classifier.py, embeddings_to_predictions.py)
* Use MINE to measure mutual information in the networks (measure_tishby.py, measure_big_nets.py, analyze_results.py)

Structure of directories and files:

#### paper code

Code from Tishby's paper

#### plots

All of the saved plots

#### saved

Saved data like Python dictionaries with results, classifier network weights, etc.

Not all present because of the big size of it.

#### tishby_data

Dataset from the Tishby's paper converted no Numpy arrays

#### tishby_plots/plots_refactored.py

Some refactored code from Tishby's paper used to show the Information Plane dynamics.

#### 2d_xor.py

Making a XOR-like dataset using the Mixture class from mixture.py

#### Tyshchuk.pdf

The actual term paper, in Russian

#### analyze_results.py

Some code to analyze and plot the results obtained in measure_big_nets.py

#### embeddings_to_predictions.py

Turn embeddings obtained in images_to_embeddings.py to generate predictions by classifiers with weights trained in train_classifiers.py

#### getting_models.py

Helper functions for constructing Keras models or getting the pretrained ones.

#### images_to_embeddings.py

Feed images from cats_vs_dogs dataset to our networks (ResNets, Inception, Xception).

#### measure_big_nets.py

Measure mutual information between input and embeddings + noise, embeddings and labels, classifier predictions and labels.

#### measure_tishby.py

Measure MI in the net from the Tishby's paper during its training.

#### mine.py

Main code for training MINE nets and getting MI estimates.

#### mine_test_from_paper.py

Recreated test of MINE performance on multivariate normal distributions like in MINE paper.

#### mine_test_infinity.py

Test MINE and AA-MINE performance on random variables with infinite MI.

#### mine_test_mixture.py

Test MINE on data obtained from Mixture class from mixture.py

#### mine_test_tishby_ds.py

Test MINE performance on the dataset form Tishby's paper.

#### mixture.py

Mixture class. Able to sample points and to measure MI using numeric integration.

#### train_classifiers.py

Train simple (one Dense(1) layer) classifiers for embeddings of pictures from cats vs dogs dataset.

#### Tyshchuk.pdf

The actual term paper.

#### utils.py

Some helper functions.

#### Полезные ссылки.md

Набор ссылок на статьи из области интереса.

