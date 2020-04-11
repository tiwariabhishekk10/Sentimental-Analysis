# Sentimental-Analysis

# Problem
The problem is to classify the messages as positive or negative sentiment message. Therefore, to solve the problem Naïve Bayes Classifier algorithm is used

# Naive Bayes Classifer
Naive Bayes classifier is a probabilistic classifier based on Bayes theorem. Naive Bayes classifier is used for both binary as well as multi-class classification states that the probability of hypothesis P(A) given some evidence P(B). The assumption for the classifier that the predictors are independent of each other. The Bayes theorem is given as: 
 
𝑃(𝐴|𝐵) =
(𝑃(𝐵|𝐴)𝑃(𝐴))*𝑃(𝐵)
 
 
 
The Bayes theorem can also be written as: 
𝑃(𝑦|𝑋) =
(𝑃(𝑋|𝑦)𝑃(𝑦))*𝑃(𝑋)
 
 
Where 𝑦 is the dependent variable with more than a single class.  The X = (x1,x2, ...., xn) be the number of predictors or independent variables.  Naïve Bayes performs better then logistic regression when assumption of independence holds.

# Analysis
The analysis and model building was data on python using package sklearn(). The Naïve Bayes classifier model aims to predict the positive or negative class of each message. The model was saved using the python package pickle().
