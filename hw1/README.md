# CS632 Deep Learning -  Assignment #1


## Part 1: Nearest Neighbor Classifier

Mimicking the API of scikit-learn.


### Conceptual questions

**1. In a Nearest Neighbor classifier, is it important that all features be on the same scale? Think: what would happen if one feature ranges between 0-1, and another ranges between 0-1000? If it is important that they are on the same scale, how could you achieve this?**

Features have to be standardized. Standardization techniques is a dense topic, but the main goal is to rescaled all feature so that they’ll have the properties of a standard normal distribution (i.e. scaling to [0, 1] range).  It's important to normalize especially when the scale of a feature is not relevant and/or misleading. And also take in consideration if feature scaling matters for the current approach, in this case, for example, the k-nearest neighbors take into account a Euclidean distance measure and that means that all features to contribute equally, therefore, scaling it's almost always necessary.

**2. What is the difference between a numeric and categorical feature? How might you represent a categorical feature so your Nearest Neighbor classifier could work with it?**

Numerical means that can be measured, categorial means that can be labeled. A common and simple method is to represent categorical features in a KNN is to transform the nominal attribute into a binary attribute. E.g.: Spam filtering (part 2).

**3. What is the importance of testing data?**
I can't think is a way to measure and improve a model performance without testing data. Even techniques as cross-validation use different sets.

**4. What does “supervised” refer to in “supervised classification”?**
Supervised means that somebody somewhere already knows the answer, output or label to expect from a testing data. The algorithm's task is to infer a function from this labeled training data so that it can learn and then predict.

**5. If you were to include additional features for the Iris dataset, what would they be, and why?**
The dataset is great for general purposes, being a toy set, I'd probably try to reduce the noise between Versicolor and Virginica by adding a new feature, so that it would possible to have higher accuracy thus more control over what's happening in your model.


## Part 2: Spam Classifier

Starting from a zip file of emails the instructor will provide. You will divide the data into
train/test, write code to extract features, run experiments, and consider the results.


### Conceptual questions

**1. What are the strengths and weaknesses of using a Bag of Words? (Tip: would this representation let you distinguish between two sentences the same words, but in a different order?)**

A bag of words representation would not let me distinguish between two sentences the same word as the context of a word is not relevant in this model. This approach will dump all words into a mathematical model, without any additional knowledge or interpretation of linguistic patterns and properties such as word order (“a good book” versus “book a good”), synonyms, spelling and syntactical variations, co-references and pronouns resolution or negations.

On the other side, this approach it's quite simple to implement as it only takes into account the occurrence of each word as a feature for training a classifier.

**2. Which of your features do you think is most predictive, least predictive, and why?**

I believe the features that are more predictive are the words with the highest weight and that is only in the emails labeled as spam. The least predictive might be common words with the highest weight. BOW itself it's not a good an approach to this problem due to the reason stated above, I'd prefer a Naive Bayes.

**3. Did your classifier misclassify any examples? Why or why not?**

The dataset doesn't have a good-enough size to train a BOW approach. True positives were part luck, part a slight similarity among spam text content.

## Prerequisites:
**Textblob Library:**
Install the "natural language processing" TextBlob library:  http://textblob.readthedocs.io/en/stable/install.html

```sh
$ pip install -U textblob
$ python -m textblob.download_corpora
```

**scikit-learn:**
```sh
pip install -U scikit-learn
```

## Constraints:

Will get rid of:
* HTML code
* XML Code
* Numbers
* Special Characters
* 1 letter words

More:

* English language only.
* Will transform word to its base form (lemma).
* Will lowercase all words.
* Do not alter the file structure of any folder within this repo.
* Run it several times: Highest accuracy reached was 76%.
