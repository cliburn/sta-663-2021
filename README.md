# STA 663 (Spring 2021) Syllabus

## Synopsis

STA 663: Computational Statistics and Statistical Computing

This course is designed for graduate research students who need to analyze complex data sets, and/or implement efficient statistical algorithms from the literature. We focus on the following analytical skills:

Functional programming in Python: Python is a dynamic programming language that is increasingly dominant in many scientific domains such as data science, computer vision and machine learning. The course will develop fluency in Python and its standard scientific scientific computing numerical and visualization packages. As most modern parallel and distributed programming libraries such as  TensorFlow emphasize use of pure functions and lazy evaluation, we will also develop the ability to program in a functional style in Python.

Statistical algorithms: Statisticians need to understand the methods they use, so that the methods can be used appropriately and extended if necessary. Using Python, we will study common numerical algorithms used in statistical model construction and fitting, starting with the basic tools for solving numerical problems, and moving on to statistical inference using optimization and simulation strategies in both frequentist and Bayesian settings. Foundational concepts from linear algebra, calculus and probability will be reviewed where needed.

Improving performance: With real-world data being generated at an ever-increasing clip, we also need to be concerned with computational performance, so that we can complete our calculations in a reasonable time or handle data that is too large to fit into memory. To do so, we need to understand how to evaluate the performance of different data structures and algorithms, language-specific idioms for efficient data processing, native code compilation, and exploit resources for parallel computing, including statistical approaches that capitalize on the high-performance libraries originally developed for deep learning.

The capstone project involves the creation of an optimized Python package implementing a statistical algorithm from the research literature.

## Learning objectives

- Develop fluency in Python, especially the functional style, for scientific computing
- Understand how common numerical algorithms used in frequentist, Bayesian and machine learning settings work
- Implement, test, optimize, and package a statistical algorithm

Note: The syllabus is aspirational and is likely to be adjusted over the semester depending on how fast we are able to cover the material.

## Administration

### Office hours 

- Cliburn: 8-10 PM MON (Scheduled upon request)
- Jiawei Chen 8-10 PM FRI
- Bo Liu 8-10 PM WED

Office hours: TBD
### Grading

- Homework 40%
- Midterm 1 15%
- Midterm 2 15%
- Project 30%

### Point range for letter grade

- A 94 - 100
- B 85 - 93
- C 70 - 84
- D Below 70

### Unit 1: Develop fluency in Python for scientific computing

#### Jupyter and Python

- Introduction to Jupyter
- Using Markdown
- Magic functions
- REPL
- Data types
- Operators
- Collections
- Functions and methods
- Control flow
- Packages and namespace
- Coding style
- Understanding error messages
- Getting help
- Saving and exporting Jupyter notebooks

#### Text 

- The string package
- String methods
- Regular expressions
- Loading and saving text files
- Context managers
- Dealing with encoding errors

#### Numerics

- Issues with floating point numbers
- The `math` package
- Constructing `numpy` arrays
- Indexing
- Splitting and merging arrays
- Universal functions - transforms and reductions
- Broadcasting rules
- Masking
- Sparse matrices with `scipy.sparse`

#### Data manipulation

- Series and DataFrames in `pandas`
- Creating, loading and saving DataFrames
- Basic information
- Indexing
- Method chaining
- Selecting rows and columns
- Transformations
- Aggregate functions
- Split-apply-combine
- Window functions
- Hierarchical indexing

#### Graphics

- Grammar of graphics
- Graphics from the group up with `matplotlib`
- Statistical visualizations with `seaborn`

#### Functional programming in Python

- Writing a custom function
- Pure functions
- Anonymous functions
- Lazy evaluation
- Higher-order functions
- Decorators
- Partial application
- Using operator
- Using `functional`
- Using `itertools`
- Pipelines with `toolz`

### Unit 2: Making code faster

#### Parallel programming

- Parallel, concurrent, distributed
- Synchronous and asynchronous calls
- Threads and processes
- Shared memory programming pitfalls: deadlock and race conditions
- Embarrassingly parallel programs with `concurrent.futures` and `multiprocessing`
- Using `ipyparallel` for interactive parallelization

#### JIT and AOT code optimization

- Source code, machine code, runtime
- Interpreted vs compiled code
- Static vs dynamic typing
- The costs of dynamic typing
- Vectorization in interpreted languages
- JIT compilation with `numba`
- AOT compilation with `cython`

#### Introduction to modern C++

- Hello world
- Headers and source files
- Compiling and executing a C++ program
- Using `make`
- Basic types and type declaration
- Loops and conditional execution
- I/O
- Functions
- Template functions
- Anonymous functions

#### Wrapping C++ for use in Python

- Using STL containers
- Using STL algorithms
- Numeric libraries for C++
- Hello world with `pybind11`
- Wrapping a function with `pybind11`
- Integration with `eigen`

### Unit 3: How common statistical algorithms work

#### Data structures, algorithms and complexity

- Sequence and mapping containers
- Using collections
- Sorting
- Priority queues
- Working with recursive algorithms
- Tabling and dynamic programing
- Time and space complexity
- Measuring time
- Measuring space

#### Solving linear equations

- Vectors and vector spaces 
- Matrices and linear mappings
- Norms, inner product, cross product
- Linear combinations and independence
- Rank, span, basis, dimensions
- Solving Ax = 0
- Solving Ax = bAx=b
- Gaussian elimination and LR decomposition
- Symmetric matrices and Cholesky decomposition
- Geometry of the normal equations
- Gradient descent to solve linear equations
- Using `scipy.linalg`

#### Singular Value Decomposition

- Change of basis
- Spectral decomposition
- Geometry of spectral decomposition
- The four fundamental subspaces of linear algebra
- The SVD
- Geometry of spectral decomposition
- SVD and low rank approximation
- Using `scipy.linalg`

#### Optimization I

- Root finding
- Univariate optimization
- Geometry and calculus of optimization
- Gradient descent
- Batch, mini-batch and stochastic variants
- Improving gradient descent
- Root finding and univariate optimization with `scipy.optim`

#### Optimization II

- Nelder-Mead (Zeroth order method)
- Line search methods
- Trust region methods
- IRLS
- Lagrange multipliers, KKT and constrained optimization
- Multivariate optimization with `scipy.optim`

#### Dimension reduction

- Matrix factorization - PCA and SVD, MMF
- Optimization methods - MDS and t-SNE
- Using `sklearn.decomposition` and `sklearn.manifold`
  
#### Interpolation

- Polynomial
- Spline
- Gaussian process
- Using `scipy.interpolate`

#### Clustering

- Partitioning (k-means)
- Hierarchical (agglomerative Hierarchical Clustering)
- Density based (dbscan, mean-shift)
- Model based (GMM)
- Self-organizing maps
- Cluster initialization
- Cluster evaluation
- Cluster alignment (Munkres)
- Using `skearn.cluster`

### Unit 4: Probabilistic Programming

#### Random numbers and Monte Carlo methods

- Working with probability distributions
- Where do random numbers in the computer come from?
- Sampling form data
- Bootstrap
- Permutation
- Leave-one-out
- Likelihood and MLE
- Using `random`,  `np.random` and  `scipy.statistics`

#### Review of Markov Chain Monte Carlo (MCMC)

- Bayes theorem and integration
- Numerical integration (quadrature)
- MCMC concepts
- Makrov chains
- Metropolis-Hastings random walk
- Gibbs sampler
- Hamiltonian systems
- Integration of Hamiltonian system dynamics
- Energy and probability distributions
- HMC
- NUTS

#### PyMC and PyStan

- Multi-level Bayesian models
- Using daft to draw plate diagrams
- Using `pymc3`
- Using `pystan`

#### TensorFlow Probability

- TensorFlow as a numerical library
- TensorFlow probability (`tfp`)

#### Probabilistic deep learning

- Building deep learning models
- Flexible distributions
- Normalizing flow

#### Bayesian deep learning

- Bayesian neural networks
- Variational inference with `tfp`
- Monte Carlo dropout