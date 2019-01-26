# MultiClass_Logistic_Regression
This model has been trained to classify handwritten digits(subset of MNIST dataset).
I have used the famous One-vs-All method to train this model.

One-Vs-All:
Suppose we have 4 classes. We train 4 independent logistic regression units one for each class as positive and other examples as negative.
Thus,
For first model, positive class- 1, negative class-(2,3,4)

For second model, positive class-2, negative class-(1,3,4)

For third model, positive class-3, negative class-(1,2,4)

For Fourth model, positive class-4,negative class-(1,2,3)

We then choose the class based on the probability of each of the models. 
Thus if first model has highest probability then example belongs to the first class.
