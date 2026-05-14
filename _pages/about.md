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

<span style="color: rgba(58, 75, 209, 0.9);">It is unrealistic to read all the materials listed below in one semester, but students must read some of them in order to finish a qualified essay. </span>

The materials are collected and reorganized mainly from:

* [H. Liu, J. Hu, Y. Li, Z. Wen, Optimization: Modeling, Algorithm and Theory (in Chinese)](http://faculty.bicmr.pku.edu.cn/~wenzw/optbook.html)
* Beck, A. *Introduction to Non-linear Optimization: Theory, Algorithm, and Applications in MALTAB*, SIAM, 2014.
* Beck, A. *First-Order Methods in Optimization*, SIAM,  2017.
* Gartner, B., He, N., and Jaggi, M. Lectures notes on *Optimization for Data Science*.
* Nesterov, Y. *Lectures on Convex Optimization*, Springer, 2018
* Nesterov, Y. Lecture notes on *Modern Optimization* in 2024 summer school at Peking University.
* Lan, G. *First-order and Stochastic Optimization Methods for Machine Learning*, Springer, 2020.

<span style="color: rgba(211, 23, 23, 0.88);">ALL THE MATERIALS ARE INTENDED FOR NON-PROFIT ACADEMIC USE. IF THEY ARE PRESENTED IMPROPERLY, PLEASE EMAIL ME TO REQUEST REMOVAL.</span>

### <span style="color: rgba(118, 24, 24, 0.88);">Class schedule</span>

|                Class time                 |               Location               |
| :---------------------------------------: | :----------------------------------: |
| 14:00---15:40, Thursday, week 1 to week 9 | N333, Teching Building, Panyu Campus |
|  10:30---12:10, Friday, week 1 to week 9  | N224, Teching Building, Panyu Campus |

### <span style="color: rgba(118, 24, 24, 0.88);">Syllabus</span>

* Lecture 1 Mathematical Preliminaries [[notes](../notes/lecture1.pdf)] 
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

* Lecture 2 Smooth Convex Functions [[notes](../notes/lecture2.pdf)] 
  * Smooth functions
  * Smooth convex functions
  * Strongly smooth convex functions

* Lecture 3 Gradient Descent [[notes](../notes/lecture3.pdf)][[python demo](../codes/gradient_descent.ipynb)][[matlab demo](../codes/run_gd_nagd.m)]
  * Vallina Gradient Descent
  * Acceralated Gradient Descent
    * Reading
      * [Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization." ICLR (2015).](https://arxiv.org/pdf/1412.6980)[**Adam, one the most frequently used optimizer for deep learning**]
      * [Loshchilov, Ilya, and Frank Hutter. "Decoupled weight decay regularization." ICLR (2019).](https://arxiv.org/abs/1711.05101)[**AdamW**]
      * [Reddi, Sashank J., Satyen Kale, and Sanjiv Kumar. "On the convergence of adam and beyond." ICLR (2018).](https://arxiv.org/pdf/1904.09237)[**Adam fails to convergent**]
      * [El Hanchi, Ayoub, David Stephens, and Chris Maddison. "Stochastic reweighted gradient descent." International Conference on Machine Learning. PMLR, 2022.](https://proceedings.mlr.press/v162/hanchi22a/hanchi22a.pdf)
      * [Keller Jordan. Muon: An optimizer for hidden layers in neural networks.](https://kellerjordan.github.io/posts/muon/)[**Optimizer for parameters in matrix form**]
  
* Lecture 4 Coordinate Descent and Conjugate Gradient Descent [[notes](../notes/lecture4.pdf)]  [[notes](../notes/lecture5.pdf)]
    * Reading
      * [Bansal, N., & Gupta, A. (2017). Potential-function proofs for first-order methods. arXiv preprint arXiv:1712.04581.](https://arxiv.org/pdf/1712.04581)[**Construction of potential functions for convergence analysis**]
      * [Karimi, H., Nutini, J., & Schmidt, M. (2016, September). Linear convergence of gradient and proximal-gradient methods under the polyak-łojasiewicz condition. In Joint European conference on machine learning and knowledge discovery in databases (pp. 795-811). Cham: Springer International Publishing.](https://arxiv.org/pdf/1608.04636)[**PL inequality for convergence analysis of gradient descent**]
      * [Shewchuk, J. R. (1994). An introduction to the conjugate gradient method without the agonizing pain.](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
      * [Nocedal, J., & Wright, S. J. (Eds.). (2006). *Numerical optimization*. 2nd Edition,  New York, NY: Springer New York. Chapter 5.](https://link.springer.com/book/10.1007/978-0-387-40065-5)

* Lecture 5 Projected Gradient Descent  [[notes](../notes/lecture6.pdf)]
  * Optimization over convex sets
  * The orthogonal projection
  * The gradient projection method

* Lecture 6 Frank-Wolfe Algorithm  [[notes](../notes/lecture7.pdf)]
  * Reading
    * [Jaggi, Martin. "Revisiting Frank-Wolfe: Projection-free sparse convex optimization." *International conference on machine learning*. PMLR, 2013.](http://proceedings.mlr.press/v28/jaggi13.pdf)
    * [Ding, Lijun, et al. "Spectral frank-wolfe algorithm: Strict complementarity and linear convergence." *International conference on machine learning*. PMLR, 2020.](http://proceedings.mlr.press/v119/ding20a/ding20a.pdf)
    * [Dvurechensky, Pavel, et al. "Self-concordant analysis of Frank-Wolfe algorithms." *International Conference on Machine Learning*. PMLR, 2020.](http://proceedings.mlr.press/v119/dvurechensky20a/dvurechensky20a.pdf)
    * [Zhou, Baojian, and Yifan Sun. "Approximate Frank-Wolfe Algorithms over Graph-structured Support Sets." *International Conference on Machine Learning*. PMLR, 2022.](https://proceedings.mlr.press/v162/zhou22i/zhou22i.pdf)

* Lecture 7 Subgradient Methods
  * Subgradient and Subdifferential [[notes](../notes/lecture8.pdf)]
  * Subgradient Method  [[notes](../notes/lecture9.pdf)]

* Lecture 8 Mirror Descent  [[notes](../notes/lecture10.pdf)]
  * Bregman divergence
  * Mirror Descent
    * Reading
      * Beck, A. *First-Order Methods in Optimization*, SIAM,  2017, Chapter 9
        * Example 9.19
        * Example 9.29
  
* Lecture 10 Conjugate Functions and Smoothing
  * Convex conjugate theory [[notes](../notes/lecture11.pdf)]
  * Smoothing techniques

* Lecture 11 Proximal operator and Proximal Gradient Descent  [[notes](../notes/lecture12.pdf)]
  * Proximal operators
  * Proximal point algorithm
  * Proximal gradient descent
  * Non-Euclidean proximal gradent methods [[notes](../notes/lecture13.pdf)]
    * Beck, A. *First-Order Methods in Optimization*, SIAM,  2017, Chapter 10
      * Example 10.76
  
* Lecture 12 Stochastic Optimization  [[notes](../notes/lecture14.pdf)]
  * Stochastic gradient descent
  * [Sample codes for reproducing Figure 13.1](../codes/stochastic_optimization) in Gartner, B., He, N., and Jaggi, M. Lectures notes on *Optimization for Data Science*. [credit to ChatGPT]
    * Reading
      * [Nitanda, Atsushi. "Stochastic proximal gradient descent with acceleration techniques." Advances in neural information processing systems 27 (2014).](https://proceedings.neurips.cc/paper_files/paper/2014/file/d554f7bb7be44a7267068a7df88ddd20-Paper.pdf)
      * [Li, Zhize, and Jian Li. "A simple proximal stochastic gradient method for nonsmooth nonconvex optimization." Advances in neural information processing systems 31 (2018).](https://proceedings.neurips.cc/paper_files/paper/2018/file/e727fa59ddefcefb5d39501167623132-Paper.pdf)
      * [Gower, Robert M., et al. "Variance-reduced methods for machine learning." Proceedings of the IEEE 108.11 (2020): 1968-1983.](https://ieeexplore.ieee.org/document/9226504)
      * [Xiao, Lin, and Tong Zhang. "A proximal stochastic gradient method with progressive variance reduction." SIAM Journal on Optimization 24.4 (2014): 2057-2075.](https://arxiv.org/abs/1403.4699)
      * 
  
* Lecture 13 Primal-Dual Algorithm
  
  * Reading
    * [Chen, Yunmei, Guanghui Lan, and Yuyuan Ouyang. "Optimal primal-dual methods for a class of saddle point problems." SIAM Journal on Optimization 24.4 (2014): 1779-1814.](https://arxiv.org/pdf/1309.5548)
    * Vandenberghe, L. (2022). Primal-dual proximal methods [Lecture slides]. ECE236C: Optimization Methods for Large-Scale Systems, University of California, Los Angeles.  https://www.seas.ucla.edu/~vandenbe/236C/lectures/pdprox.pdf
    * Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for convex problems  with applications to imaging. Journal of Mathematical Imaging and Vision, 40, 120–145.  https://doi.org/10.1007/s10851-010-0251-1
    * Ryu, E. K., & Boyd, S. (2016). A primer on monotone operator methods. Applied and Computational Mathematics, 15(1), 3–43.
    * Ryu, E. K., & Yin, W. (2022). Large-scale convex optimization: Algorithms & analyses via  monotone operators. Cambridge University Press. https://doi.org/10.1017/9781009160865 [Chapter 3]
    * Chambolle, A., & Pock, T. (2016). An introduction to continuous optimization for imaging.  Acta Numerica, 25, 161–319.
          https://doi.org/10.1017/S096249291600009X [Section 5]
  
* Advanced Topics
  
  * Low-Rank Adaptation(LoRA) for SFT [[notes](./notes/lora.pdf)]
  * Muon
    * Bernstein, J.  [Deriving Muon](https://jeremybernste.in/writing/deriving-muon)
    * Liu, Jingyuan, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin et al. "Muon is scalable for llm training." *arXiv preprint arXiv:2502.16982* (2025).
    * Jordan, K., et al. [Muon: An optimizer for hidden layers in neural networks.](An optimizer for hidden layers in neural networks.)
    * Hugging Face. [Understanding the Muon Optimizer: Theory and Implementation](https://huggingface.co/datasets/bird-of-paradise/muon-tutorial)
    * Lecture 8. [EPFL Course - Optimization for Machine Learning - CS-439](https://raw.githubusercontent.com/epfml/OptML_course/main/slides/lecture08.pdf)
    
  * Trajectory analysis
  

### <span style="color: rgba(118, 24, 24, 0.88);">Homework </span>

Homework will be released from time to time.

| Date       | Assignment                                                   | Remark |
| ---------- | ------------------------------------------------------------ | ------ |
| 2026-03-12 | Reading: K. Wen, D. Hall, T. Ma, and P. Liang. Fantastic pretraining optimizers and where to find them. arXiv preprint arXiv:2509.02046, 2025. |        |
|            |                                                              |        |
|            |                                                              |        |
|            |                                                              |        |
|            |                                                              |        |
|            |                                                              |        |
|            |                                                              |        |
|            |                                                              |        |
|            |                                                              |        |

