This is the final project for SJTU's Statistical Learning class. I scored in the top 5 for the Kaggle competition (out of  around 200 students)

I initially used R and the Caret framework to explore different classical ML algorithms, however in the end what worked best was using semi-supervised learning techniques (like self-learning using EM-family algorithms). These algorithms  try to find structure to the data by also the using test set (which was around ten times larger). 

No actual RStudio notebook is provided. However, all the procedures used are in the procedures.r library. If you wish to use the code, have a look at the readme.html file which explains how to use the API it exposes.  First, you need to compute the PC feature space and save it somewhere:

```R
make_pca()
```

Then, for example to do self-learning:

```R
makeSelfLearningModel("LeastSquaresClassifier",1000)
```

It's very straightforward, all is explained in the readme.html file.