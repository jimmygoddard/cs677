Jimmy Goddard
MET CS 677 Final Project
04/29/19

# Technical details

Python Libraries used: matplotlib, numpy, pandas, seaborn, and sklearn

To run the code after installing the required libraries, either load it into your IDE and run it or using Python
at the command line:

  > python FinalProject.py

The dataset is included in this zip in the dataset directory.  The original zip downloaded from Kaggle is also
in the dataset directory.  To read about or download the dataset yourself, you can go to
https://www.kaggle.com/fivethirtyeight/fivethirtyeight-comic-characters-dataset

# Project Description

For my final project I explored a categorical data set from Kaggle containing information on characters from
Marvel and DC comics.

I was mainly interested in how many characters were added per year and explored the gender and sexuality
distributions of these new characters.

A histogram of total characters added per year across both publishers showed that there was an early peak in the 1940's
when the companies were founded and then a larger peak in the 1990's through the 2010's.

A histogram of characters added per by broken out by publisher showed that DC added a larger proportion of its characters
later, in the 1990's through 2010's.  Marvel had a larger proportion of its characters added right around when it was
founded and had a more even output of new characters through the peak of the 1990's to the 2010's.

While male and female genders far outnumber other genders, there is some representation for non-binary genders with,
to me, a surprising number of non-binary gendered heroes prior to 1990.

Not including straight heroes, we can see that female characters tended to have more diversity for gender or sexual
minority categories.  Male heroes in the gender or sexual minority category tended to be homosexual.

Non-binary genders comprised an insignificant amount of new heroes created per year.
Female new heroes also account for very little of the overall number of new heroes.

To do classification, I trimmed the dataset to only include male or female genders with the purpose of being able to
classify the gender of a character given the identification, alignment, eye color, hair color, gender or sexual minority
classification, whether they are alive or dead, number of appearances, year of first appearance, and publisher.

I implemented a variety of individual classifiers; kNN, naive bayesian, logistic regression, support vector machine,
and decision tree; to do classification.  I also employed two ensemble methods: random forest and a voting classifier.

I also implemented a kMeans clustering model, though its not clear to me what relevance that has for gender and sexuality
analysis in this dataset.

For kNN kMeans, and the random forest, I created a series of models with different hyperparameters to find the optimal
values for this dataset.  I ended up with k = 9 for kNN, k = 16 for kMeans, and 2 estimators with a max depth of 1 for
the random forest.  Not only is it surprising that the random forest has such small hyperparameters, it also indicates
to me that there are only a few important features in the dataset.  It would have been enlightening to do more formal
statistical analysis to determine where there were dependencies in the features.

Following is a table showing the accuracy of each of the classifiers.

                model  accuracy
        Decision Tree  0.659408
                  kNN  0.730046
                   NB  0.746693
                  SVM  0.747233
  Logistic Regression  0.750742
    Voting Classifier  0.755872

The Voting Classifier was the most interesting to me.  It was comprised of kNN, logistic regression, naive bayesian,
SVM, and decision tree classifiers.  I used the same hyperparameters for each of the models as I did when I used them
individually.  I split the dataset 50/50 into training and testing partitions.

As I mentioned above, it would have been interesting and useful to do a more formal statistical analysis of the data
during discovery.  At the very least, I may have been able to eliminate highly dependent features to save computation
time.

I also would have liked to implement a neural network to do classification, but did not have the time to do so.

I did enjoy using Python and its family of libraries to do the classification and exploration.  I've done a fair amount
of data analysis in R and have been looking forward to using Python to do similar work.  I have a long history of
familiarity with Python but have never used it for scientific computing.  It did not disappoint.  It would have been
interesting to explore more visualization libraries too.  I like matplotlib and seaborn, but there are others like Bokeh
that provide very interesting ways of interacting with the data.

Also, I'm a huge fan of comic books, Marvel especially, and it was a fun process to look at data scraped from the
wiki's.  It wasn't surprising that females as well as gender and sexual minorities were severely underrepresented in the data.
Maybe that will change as more movies similar to Captain Marvel and Wonder Woman influence younger fans of super heroes.