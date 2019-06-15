#Reference

## Feature Description

* 10000維中，前5000維是所謂的mean squared displacement (MSD) as a function of time interval，5000個維度分別代表5000個不同time interval下，軌跡中位移的second moment的平均。The 1st to the 5000th represent the increasing time intervals, from the shortest time interval (between consecutive steps) to the longer interval (between 5000 steps).
* MSD是用來分析運動是否碰到阻礙常用的方法。在完全沒有碰到阻礙的情況下，MSD應該是linearly proportional to the time interval。在我們的例子中，有兩個因素（分別是fMB 以及 mesh confinement）都會使MSD偏離一條直線。理想上，我們希望從MSD的彎曲程度和方式，來了解軌跡中的資訊。
* 剩下的5000維是所謂的velocity autocorrelation (VAC)，它是由50組VAC data串連組成，每組有100個維度。VAC也是分析擴散運動型態的常用方法之一，他的概念是在軌跡中找不同time interval時，速度的correlation。VAC的100個維度，代表100個不同的time interval。另外，我們也調整計算速度的方法，從非常瞬間的速度，至較為平均的速度，總共挑了50個計算速度的方式。由此構成了50x100 = 5000維度的資訊。
* 總合來說，feature matrix是有兩組常用的統計分析方法之結果。一般的物理方法，會試著推導出這兩個統計分析方法之解析解，然後試著fitting，找到決定軌跡的未知變數（in our case, mesh size L, alpha, transmission probability）。在我們的例子中，推導出解析解非常困難，然而我相信這5000維的feature matrix中包含了很多的消化過的資訊，希望足夠machine learning來估計那三個未知數。
* 最後一提，以上總共10,000 (1e4)維的feature是從總共100,000 (1e5) steps的軌跡計算而來。我們也考慮過直接把軌跡位置的data當作feature（完全不做額外處理或分析），但資料量太大，所以後來改成現行的做法（資料量還是蠻大）
* [Supplementary : VAC](http://people.virginia.edu/~lz2n/mse627/notes/Correlations.pdf)

## Useful Links

### XGboost
* [XGboost - Official Document](https://xgboost.readthedocs.io/en/latest/index.html)
* [XGboost - iT邦幫忙](https://ithelp.ithome.com.tw/articles/10205409)
* [Kaggle機器學習競賽神器XGBoost介紹 - Medium](https://medium.com/jameslearningnote/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC5-2%E8%AC%9B-kaggle%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AB%B6%E8%B3%BD%E7%A5%9E%E5%99%A8xgboost%E4%BB%8B%E7%B4%B9-1c8f55cffcc)
* [XGboost 原理 - 知乎](https://www.zhihu.com/question/58883125)
* [XGBoost – A Scalable Tree Boosting System：Kaggle 競賽最常被使用的演算法之一](https://medium.com/@cyeninesky3/xgboost-a-scalable-tree-boosting-system-%E8%AB%96%E6%96%87%E7%AD%86%E8%A8%98%E8%88%87%E5%AF%A6%E4%BD%9C-2b3291e0d1fe)
* [XGboost - 調參指南](https://blog.csdn.net/han_xiaoyang/article/details/52665396)

## Training skills
* [A bunch of tips and tricks for training deep neural networks](https://towardsdatascience.com/a-bunch-of-tips-and-tricks-for-training-deep-neural-networks-3ca24c31ddc8)
* [Estimating an Optimal Learning Rate For a Deep Neural Network](https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0)
* [Train your deep model faster and sharper — two novel techniques](https://hackernoon.com/training-your-deep-model-faster-and-sharper-e85076c3b047)
* [Deep Learning Tips and Tricks](https://towardsdatascience.com/deep-learning-tips-and-tricks-1ef708ec5f53)
* [Deep Learning Tips and Tricks cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-deep-learning-tips-and-tricks)
* [DL tricks - github](https://github.com/Conchylicultor/Deep-Learning-Tricks#training)
* [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)

## Feature Engineering
* [Feature Engineering 特徵工程中常見的方法](https://vinta.ws/code/feature-engineering.html)
* [特征工程方法综述](https://cloud.tencent.com/developer/article/1005443)
* [Feature Engineering 相關文章推薦](https://medium.com/@drumrick/feature-engineering-%E7%9B%B8%E9%97%9C%E6%96%87%E7%AB%A0%E6%8E%A8%E8%96%A6-b4c2aaffe93d)
* [Fundamental Techniques of Feature Engineering for Machine Learning](https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114)
* [Rare Feature Engineering Techniques for Machine Learning Competitions](https://medium.com/ml-byte/rare-feature-engineering-techniques-for-machine-learning-competitions-de36c7bb418f)
* [Feature Engineering: What Powers Machine Learning](https://towardsdatascience.com/feature-engineering-what-powers-machine-learning-93ab191bcc2d)
