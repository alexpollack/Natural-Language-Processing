# Natural-Language-Processing
Implementing various statistical method in natural language processing

# Fasttext Word Embeddings
This file uses the fast text trained unsupervised model to train word embeddings. The embeddings were learned for a file containing 86965 words for embeddings of dimension size 100. The result was embeddings that when evaluted scored an accuracy of 0.65. 
The embedding were also learned with mixed set of values for dimension size and the amount of data used. The best and inital embeddings used 100% of the data and the dimension was 100. Embeddings were also learned using just 50% (resulting in 62443 words) and 25% (resulting in 42052 words)of the total data (% by line count of the initial data file). The embeddings were also learned with dimension size set to be 50 and 25. Between these settings, a total of 9 sets embeddings were learned and as expected the best results came from the largest amount of data (100%) and the largest dimension (100), while the lowest scoring was the 25%/25dim set of embeddings.
The result of the experiments can be seen in the following chart, and full score info in the table:
### Scores vs. Dimension Size for 100%, 50%, 25% of Training Data
###### Orange: 100%, Yellow: 50%, Blue: 25%
![alt text](/images/graph.png "Scores vs. Dimension size")

### Full Score Information
![alt text](/images/table.png "Scores for all sets of embeddings")


![alt text](/images/embeddings.png "Scores vs. Dimension size")


# Document Tagging Using Logistic Regression with Gradient Ascent
For a given set of documents (here a set of words and phrases), logistic regression was used to classify the phrases into the five classes: drug, person, place, movie, and company. In order to learn training weights, gradient ascent was implemented to adjust the weights for each class. In just 17 iterations of gradient ascent, a score of 0.60 was earned on just 1/4 of the total data set provided.
