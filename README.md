# Slow Payers

## Introduction
In this project I trained a perceptron neural network to predict if a new client will be a slow payer. This example can be a good starting point for data scientist who are begining to learn perceptron, the most common neural network algorithm.

In this reposirtory you may find:

* README.md: This file.
* slow_payers.py: Python script with the normalization, segregation, training and testing processes.
* users.xlsx: A data set with 1000 rows which each one represents a client. There are more columns than the used by me in this prediction so maybe you can use othe columns to obtain different results.

For this prediction I used 3 input variables:
* Quantity: The total quantity of the loan.
* Labor seniority: The time the client has worked in their current job (in months)
* Salary ratio: The percentage of the Monthly income vs the monthly loan payment.

All variables were normalized in the script.

## The neural network

![Perceptron neural network](/images/Neural_network.jpeg)

The output of the nn is only one boolean for if the person will be a slow payer or not: if 0 no, if 1 yes.

The nn is using 6 hidden neurons for the pattern finding.

I used a random sample of 700 clients to train the nn. The other 300 clients were used to test the accuracy of the trained nn prediction. Since my initial weights are randomized, the final results may vary. You can expect 80% - 95% accuracy.

My default patterns are:
* Alpha = 0.05
* a = 0.05
* Delta o = 0.0001
* L(H) = 6

Any question you can contact me through here or:
* [Twitter](https://twitter.com/elviajeligero)
* [LinkedIn](https://www.linkedin.com/in/miguel-solis-52381a24/)
