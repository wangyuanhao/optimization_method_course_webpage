---
permalink: /
title: " "
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
This course *Optimization Methods* is oriented for graduate students from applied statistics at the department of mathematics, Jinan University. It emphasizes on theoretical understandings of modern optimization methods for data science, particularly the first-order methods,  since implemtentation is easily accessed via LLMs (e.g. DeepSeek, ChatGPT, e.t.c.).

 <a href="https://wangyuanhao.github.io" style="text-decoration:none;color:purple">**Instructor: Weiwen Wang(王伟文)**</a>

The materials are collected and reorganized mainly from:

* [H. Liu, J. Hu, Y. Li, Z. Wen, Optimization: Modeling, Algorithm and Theory (in Chinese)](http://faculty.bicmr.pku.edu.cn/~wenzw/optbook.html)
* Beck, A. *Introduction to Non-linear Optimization: Theory, Algorithm, and Applications in MALTAB*, SIAM, 2014.
* Beck, A. *First-Order Methods in Optimization*, SIAM,  2017.
* Gartner, B., He, N., and Jaggi, M. Lectures notes on *Optimization for Data Science*.
* Nesterov, Y. *Lectures on Convex Optimization*, Springer, 2018
* Nesterov, Y. Lecture notes on *Modern Optimization* in 2024 summer school at Peking University.
* Lan, G. *First-order and Stochastic Optimization Methods for Machine Learning*, Springer, 2020.

<span style="color: rgba(58, 75, 209, 0.9);">It is unrealistic to read all the materials listed below in one semester, but students must read some of them in order to finish a qualified essay. </span>

### <span style="color: rgba(118, 24, 24, 0.88);">Syllabus</span>

* Lecture 1 Mathematical Preliminaries [notes] 
  * Optimization in Data Science
  * Inner Products and Norms
  * Differentiability
  * Optimiality Conditions for Unconstrained Optimization 
  * Attainable of Minima 
    * Reading
      * [Domingos, Pedro. "A few useful things to know about machine learning." Communications of the ACM 55.10 (2012): 78-87.](https://doi.org/10.1145/2347736.2347755) 
        [**A classic paper presents the practical philosophy of machine learning.**]

      * [Cai, Jian-Feng, Emmanuel J. Candès, and Zuowei Shen. "A singular value thresholding algorithm for matrix completion." SIAM Journal on optimization 20.4 (2010): 1956-1982.](https://ww3.math.ucla.edu/camreport/cam08-77.pdf)
        [**Appoximated minimization of nuclear norm**]

      
      * [Recht, Benjamin, Maryam Fazel, and Pablo A. Parrilo. "Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization." SIAM review 52.3 (2010): 471-501.](https://arxiv.org/abs/0706.4138)
        [**Translate rank minimization to nuclear norm minimization**]

      * [Vidal, René, Ma, Yi, and S.S. Sastry. "Generalized Principal Component Analysis". Springer New York, NY. 2016. Chapter 2](https://link.springer.com/book/10.1007/978-0-387-87811-9)
        [**Application of L21 norm and nuclear norm minimization**]

* Lecture 2 Smooth Convex Functions [notes]
  * Smooth functions
  * Smooth convex functions
  * Strongly smooth convex functions

  * Definition of Smooth Convex Functions and Their Properties
* Lecture 3 Gradient Descent [notes]
  * Vallina Gradient Descent
  * Acceralated Gradient Descent
* Lecture 4 Coordinate Descent and Conjugate Gradient Descent
    * Reading
      * [Shewchuk, J. R. (1994). An introduction to the conjugate gradient method without the agonizing pain.](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
      * [Nocedal, J., & Wright, S. J. (Eds.). (2006). *Numerical optimization*. 2nd Edition,  New York, NY: Springer New York. Chapter 5.](https://link.springer.com/book/10.1007/978-0-387-40065-5)

* Lecture 5 Projected Gradient Descent
  * Optimization over convex sets
  * The orthogonal projection
  * The gradient projection method

* Lecture 6 Frank-Wolfe Algorithm
  * Reading
    * [Jaggi, Martin. "Revisiting Frank-Wolfe: Projection-free sparse convex optimization." *International conference on machine learning*. PMLR, 2013.](http://proceedings.mlr.press/v28/jaggi13.pdf)
    * [Ding, Lijun, et al. "Spectral frank-wolfe algorithm: Strict complementarity and linear convergence." *International conference on machine learning*. PMLR, 2020.](http://proceedings.mlr.press/v119/ding20a/ding20a.pdf)
    * [Dvurechensky, Pavel, et al. "Self-concordant analysis of Frank-Wolfe algorithms." *International Conference on Machine Learning*. PMLR, 2020.](http://proceedings.mlr.press/v119/dvurechensky20a/dvurechensky20a.pdf)
    * [Zhou, Baojian, and Yifan Sun. "Approximate Frank-Wolfe Algorithms over Graph-structured Support Sets." *International Conference on Machine Learning*. PMLR, 2022.](https://proceedings.mlr.press/v162/zhou22i/zhou22i.pdf)

* Lecture 7 Subgradient Methods
  * Subgradient and Subdifferential
  * Subgradient Method

* Lecture 8 Mirror Descent
  * Bregman divergence
  * Mirror Descent

* Lecture 10 Conjugate Functions and Smoothing
  * Convex conjugate theory
  * Smoothing techniques

* Lecture 11 Proximal operator and Proximal Gradient Descent
  * Proximal operators
  * Proximal point algorithm
  * Proximal gradient descent
  * Non-Euclidean proximal gradent methods

* Lecture 12 Stochastic Optimization
  * Stochastic gradient descent
  * Stochastic proximal gradent descent 

* Lecture 13 Nonconvex Optimization
  * Trajectory analysis
  * Escaping saddle points

* Lecture 15 Primal-Dual Algorithm
  * Reading
    * [Chen, Yunmei, Guanghui Lan, and Yuyuan Ouyang. "Optimal primal-dual methods for a class of saddle point problems." SIAM Journal on Optimization 24.4 (2014): 1779-1814.](https://arxiv.org/pdf/1309.5548)

* Lecture 14 Multi-objective Optimization and Its Applications on Multi-task Learning
  * Reading
    * [Pardalos, P.M.,  Žilinskas, A.,  Žilinskas, J.  *Non-Convex Multi-Objective Optimization*, Chapter 1-2, Springer, 2017.](https://link.springer.com/book/10.1007/978-3-319-61007-8) 

<span style="color: rgba(90,90, 90,0.5);">History</span>
======
* <span style="color: rgba(0,0,128,0.9);">[2024-09-21] Create this webpage.</span>

  
