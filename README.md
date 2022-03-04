# Isolation Forest Anomaly Detection - Identify Outliers

![Introduction](https://raw.githubusercontent.com/youngdataspace/K-Means-Clustering-Fixed-Effects/master/Introduction.gif)

## Outline
In this notebook, we will discuss:
- Why and how to look for outliers.
- How Isolation Forest works.
- The benefits and drawbacks of Isolation Forest.
- The implementation of Isolation Forest.

If you have any comments or suggestions, email me at y.s.yoon@berkeley.edu.

Let's get started!

## Introduction
In <a href="https://github.com/youngdataspace/treat-outliers/blob/main/detect_outliers.ipynb">this notebook</a>, I detect outliers using the Isolation Forest method. I use US public firm data, which are also used in my UC Berkeley Haas PhD Dissertation (<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3689446">Yoon 2022</a>). Although I detect ouliers to treat them before I conduct the analysis on the data, the anomaly detection technique can be applied to many business settings such as detecting fradulent credit card spending.

Figure 1 shows US public firms' characteristics shown in 2-dimentions. The goal of this notebook is to detect outliers as shown in red in Figure 2.

## Why and how to look for outliers
Many machine learning algorithms and regression models are suseptibale to outliers. Unless they are properly taken care of, the inferences obtained from statistical models may not be generalizable. 

There are many popular methods to detect outlikers, namely, the Z-Score and Interquartile Range methods. These methods are effective when the underlying data follows a normal distribution. However, if the data is not normally distributed, then these methods may incorrectly classify normal observations as outliers. On the other hand, Isolation Forest are non-parametic, which just simply means that we don't have to make assumptions about how the underlying data is distributed.

Furthermore, the Z-Score and Interquartile Range methods identify at the variable level. If you have reason to believe that multiple variables interact with eachother and create outliers, these methods will not be able to detect those outliers. For example, an SAT score of 1350/1600 (90th percentile) does not seem to be an outlier by itself. However, if we introduce antother dimention, age, and find that a 12 year old got 1350/1600, then this observation is likely an outlier. In contrast, Isolation Forest detects outliers at multidimensional space. 

## Isolation Forest
Isolation Forest is a tree ensemble method of detecting anomalies first proposed by <a href="https://ieeexplore.ieee.org/abstract/document/4781136?casa_token=A5ZM3TQZHhsAAAAA:DPITalJ8ZZ-5KnuBufXLZkFg6fsICEyyi0vfXmuGejd8gFtAldJ2ZFuS0JUoBAS8GPoF0JG5Kg">Liu, Ting, and Zhou (2008)</a>. Unlike other other methods that first profiles normal points to identify anomalies, Isolation Forest explicitly isolates anomalies.

Anomalies have two characteristics. They are distanced from nomal points and there are only a few of them. The Isolation Forest algorithm exploits these two characteristics. More specifically, here is how it works.
- Randomly select a feature
- Selec
on that is built on the basis of decision trees, just lie random forests, whereby trees are partitioned by first randomly selecting a feature and then selecting a random split value between the maximum and minimum value of the selected feature

The core of algorithm is to "isolate" outliers by creating decision tree over random atributes.
Similarly to Random Forest, it is built on an ensemble of binary (isolation) trees
Similar to random forest
The main idea, which is different from other popular outlier detection methods, is that isolation forecst explicityl identifies anomalies instead of profiiling normal data points.

outliers are less frequent than regular observations and are different from them in terms of values (they lie futher away from the regular observations in the feature space). That is why using random partitioning they should be identified closer to the root of the tree. Shorter average path length i.e., the number of edges an observation must pass in the tree giong from the root to the terminal noe with fewer splits necdssary

shows a figure that makes more splits
this guy also shows an anomaly score equation
https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e
score close to 1 indicates anomalies
score much smaller than 0.5 indicates nomal observations
if all scores are close to 0.5 then the entire sample does not seem to have clearly distinct anomalies

This guy generates outliers using code. Use this to visualize the explanation

Isolation forest is a tree ensemble method that is built on the basis of decision trees, just lie random forests, whereby trees are partitioned by first randomly selecting a feature and then selecting a random split value between the maximum and minimum value of the selected feature. In principle, outliers are less frequent than regular observations and are different in terms of their value. Thus by using random partitions, they should be identified as closer to the root of the tree with fewer splits necessary.

The strategy that Isolation Forest follows is that a point is easily separable from other is an outlier.

It selects one feature randomly at a time to make a split or cut the branches and uses the random threshold value that exists b/w the range of min (feature value) & max (feature value). For example the ragne of feature value is [20, 100] then the threhold values are selected randomly within range

as sown in the above figure many plits are required to segregate the normal data point in the left image.
In other words, we can say that we need to traverse till the end of three depth to find this element. Tor,
On the other hand in the right image, point Xo easily isolated from other points with only the initial 3 splits thus Isolation will mark it as an Outlier.



## Benefits and drawbacks of using Isolation Forest
Does not assume normal distribtution
It can be scaled up to handle large, high-dimensional datasets
need to pick the percentage of anomalies
Not good (post later)
https://medium.datadriveninvestor.com/lets-find-some-outliers-with-isolation-forest-4ed22175a8d3
Capacity to scale up to handel extremely large data size and high dimentional problems with a large number of irrelevant atributes

small sizes produce better
Less computational effort and low membory requirement -> utilizes no distance or density measures to detect anomalies.

n algorithm which has a linear time complexity with a low constant and a low memory requirement.

## Implementation.
One thin worth noting is the contamination parameter, which specifies the percentage of observations we blieive to be outliers (default value is 0.1).

The algorithm requres us to specify the contamination parameter which tells the algorithm how much of the data is expected to be anomalous.

## Medium ##
In my medium article
Add an outlier picture? GIF?



## 봐야할것 ##
https://medium.datadriveninvestor.com/lets-find-some-outliers-with-isolation-forest-4ed22175a8d3
https://github.com/sahandha/eif
https://ieeexplore.ieee.org/document/8888179

## 본것 ##
https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e
https://towardsdatascience.com/multi-variate-outlier-detection-in-python-e900a338da10
https://medium.com/codex/isolation-forest-outlier-detection-simplified-5d938548bb5c
