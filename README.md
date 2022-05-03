# Isolation Forest Anomaly Detection - Identify Outliers

![Introduction](https://github.com/youngdataspace/Detect-Outliers-Using-Isolation-Forest/blob/main/GIF%20Intro.gif?raw=true)

## Outline
In <a href="https://github.com/youngdataspace/treat-outliers/blob/main/detect_outliers.ipynb">this notebook</a>, I explain and implement Isolation Forest. My goal is to explain using plain English so that non-technical readers can understand the algorithm. 

This notebook includes the following topics:
- Why and how to look for outliers.
- How Isolation Forest works.
- The benefits and drawbacks of Isolation Forest.
- The implementation of Isolation Forest.

If you have any comments or suggestions, email me at y.s.yoon@berkeley.edu.

Let's get started!

## Introduction
I detect outliers using the Isolation Forest method. I use US public firm data, which are also used in my UC Berkeley Haas PhD Dissertation (<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3689446">Yoon 2022</a>). Although I detect anomalies (outliers) to treat them before I conduct analyses on the data, the anomaly detection technique can be applied to many business settings, such as detecting fraudulent credit card spending.

Figure 1 shows US public firms' features (characteristics) in 2-dimensions. The goal of this notebook is to detect outliers, as shown in red in Figure 2.

<p align="center">
  <img src="https://github.com/youngdataspace/treat-outliers/blob/main/Figure%201.png" width=80% height=80%>
  <img src="https://github.com/youngdataspace/treat-outliers/blob/main/Figure%202.png" width=80% height=80%>
</p>

## Why and how to look for outliers
Many machine learning algorithms and regression models are susceptible to outliers. An outlier is a data point that significantly deviates from other points. Unless they are properly taken care of, the inferences obtained from statistical models conducted on the data may not be useful.

There are many popular methods to detect outliers, namely, the Z-Score and Interquartile Range methods. These methods are effective when the underlying data follows a normal distribution (a distribution where most data points are closer to the mean and become less frequent as the distance to the mean increases). However, if the data is not normally distributed, then these methods may incorrectly classify normal observations as outliers. On the other hand, the Isolation Forest method is non-parametric, which simply means that we don't have to make assumptions about how the underlying data is distributed.

Furthermore, the Z-Score and Interquartile Range methods identify at the variable level. If you have reason to believe that multiple variables interact with each other and create outliers, these methods will not be able to detect those outliers. For example, an SAT score of 1350/1600 (90th percentile) does not seem to be an outlier by itself. However, if we introduce another dimension, age, and find that a 12-year-old got 1350/1600, then this observation is likely an outlier for a sub-sample of 12-year-olds. Unlike single-variable outlier detection methods, Isolation Forest detects outliers at multidimensional space.

## Isolation Forest
Isolation Forest is a tree ensemble method of detecting anomalies first proposed by <a href="https://ieeexplore.ieee.org/abstract/document/4781136?casa_token=A5ZM3TQZHhsAAAAA:DPITalJ8ZZ-5KnuBufXLZkFg6fsICEyyi0vfXmuGejd8gFtAldJ2ZFuS0JUoBAS8GPoF0JG5Kg">Liu, Ting, and Zhou (2008)</a>. Unlike other methods that first try to understand the normal points and classify abnormal points as anomalies, Isolation Forest explicitly isolates anomalies.

Anomalies have two characteristics. They are distanced from normal points and there are only a few of them. The Isolation Forest algorithm exploits these two characteristics. 

#### Plain English
Isolation Forest randomly cuts a given sample until a point is isolated. The intuition is that outliers are relatively easy to isolate. Take a look at the following GIF.

<p align="center">
  <img src="https://github.com/youngdataspace/Detect-Outliers-Using-Isolation-Forest/blob/main/GIF%20Outlier%20Split.gif?raw=true" width=80% height=80%>
</p>

It took 4 times to randomly cut the sample and isolate the red point, which is clearly an outlier.

Now, take a look at the next GIF, which attempts to cut the sample until the yellow point (normal point) is isolated.

<p align="center">
  <img src="https://github.com/youngdataspace/Detect-Outliers-Using-Isolation-Forest/blob/main/GIF%20Normal%20Point.gif?raw=true" width=80% height=80%>
</p>

This time, the algorithm took a lot more cuts.

As you can infer from the above, a data point is likely an outlier if it can be isolated only with a few random sample cuts. 

#### Step by Step
Here are the steps involving the Random Forest algorithm.

First, the algorithm creates an isolation tree by going through the following steps:<br>
[1] Randomly select a sub-sample (Sci-kit learn's default: 100 instances/data points)<br>
[2] Select a point to isolate.<br>
[3] Randomly select a feature (i.e., variable) from the set of features X.<br>
[4] Randomly select a threshold between the minimum and the maximum value of the feature x.<br>
[5] If the data point is less (greater) than the threshold, then it flows through the left branch of the tree (right). In other words, define the new minimum (maximum) of the range to the threshold for the next iteration.<br>
[6] Repeat steps 3 through 5 until the point is isolated or until a pre-defined max number of iterations is reached.<br>
[7] Record the number of times the steps 3 through 5 were repeated.

Prediction process: Isolation Forest is created by computing the following score based on a collection of trees (like 100 trees).

<p align="center">
  <img src="https://github.com/youngdataspace/treat-outliers/blob/main/Equation.JPG" width=20% height=20%>
</p>

where E[h(x)] is the average number of successful iterations for firm x and c(n) is the average iterations for unsuccessful iterations.

## Benefits and drawbacks of using Isolation Forest
#### Benefits
As I noted above, Isolation Forest does not assume normal distribution and is able to detect outliers at a multi-dimensional level. More importantly, Isolation Forest is computationally efficient: the algorithm has a linear time complexity with a low constant and a low memory requirement. Therefore, it scales well to large data sets.

According to <a href="https://ieeexplore.ieee.org/abstract/document/4781136?casa_token=A5ZM3TQZHhsAAAAA:DPITalJ8ZZ-5KnuBufXLZkFg6fsICEyyi0vfXmuGejd8gFtAldJ2ZFuS0JUoBAS8GPoF0JG5Kg">Liu, Ting, and Zhou (2008)</a>, Isolation Forest performs better than Random Forest, especially in large data sets.

#### Drawbacks
As I will discuss more in the implementation step, the Isolation Forest algorithm requires us to pick the percentage of anomalies in the dataset. Thus, we need to have at least some idea of the proportion of anomalies in our data.

Second, axis-parallel splits create some artificial normal regions. I won't go into details, but this issue is addressed by the follow-up study <a href="https://ieeexplore.ieee.org/document/8888179">Hariri, Kind, and Brunner (2021)</a>. And here are more resources: <a href="https://github.com/sahandha/eif">GitHub</a> and <a href="https://medium.datadriveninvestor.com/lets-find-some-outliers-with-isolation-forest-4ed22175a8d3">blog</a>. I will post a notebook on this topic when I get a chance.

## References
[1] <a href="https://ieeexplore.ieee.org/abstract/document/4781136?casa_token=A5ZM3TQZHhsAAAAA:DPITalJ8ZZ-5KnuBufXLZkFg6fsICEyyi0vfXmuGejd8gFtAldJ2ZFuS0JUoBAS8GPoF0JG5Kg">Liu, Ting, and Zhou (2008) Isolation Forest</a><br>
[2] <a href="https://quantdare.com/isolation-forest-algorithm/">Fuertes (2018) Isolation forest: the art of cutting off from the world</a><br>
[3] <a href="https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e">Lewinson (2018) Outlier Detection with Isolation Forest</a><br>
[4] <a href="https://www.analyticsvidhya.com/blog/2021/07/anomaly-detection-using-isolation-forest-a-complete-guide/">Akshara (2021) Anomaly detection using Isolation Forest - A Complete Guide</a>

