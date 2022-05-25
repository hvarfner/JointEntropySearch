# Joint Entropy Search for Maximally-Informed Bayesian Optimization

This is the official repository for Joint Entropy Search for Maximally-Informed Bayesian Optimization. We developed our code by building on Max-value Entropy Search (Wang and Jegelka, 2017) which in turn built on Predictive Entropy Search (Hernandez-Lobato et al., 2014), which was developed upon GPstuff (Vanhatalo et al., 2013). To keep the repository as trim and clean as possible, some of the examples and methods from previous work have been removed.

## System Requirements
All code has been tested with MATLAB 2021a. After installing the conda environment provided in `jes.yml`, add the required libraries by 

``` 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/conda/envs/jes/lib
``` 

and everything should be up and running. All experiments are run in Python3.7+, so a working such installation is necessary, and is not provided through the conda environment (which is unfortunately required to be in Python2).


While the required mex file is included, it may need to be re-compiled. To do so, do:

```
cd utils
mex chol2invchol.c -lgsl -lblas
```

Lastly, to run experiments, do one of the following:

#### For GP sample tasks (where the hyperparameters of the surrogate model are fixed):
```
matlab -nodisplay -nosplash -nodesktop -r "gp_task(path, seed, approach, dim);exit;"
```
Where `path` is the path to the experiment to run, `seed` is the seed to run, `approach` is the acquisition function in question, and `dim`, somewhat exessively, is the dimensionality of the problem. For example, one can run:

```
matlab -nodisplay -nosplash -nodesktop -r "gp_task('gp/gp_2dim.py', 42, 'JES', 2);exit;"
```


#### For all other tasks, use `synthetic_task`:
```
matlab -nodisplay -nosplash -nodesktop -r "synthetic_task(path, seed, approach, dim);exit;"
```

So to run Hartmann (6D):
```
matlab -nodisplay -nosplash -nodesktop -r "synthetic_task('synthetic/hartmann6.py, 37, 'MES', 6);exit;"
```

Every experiment is automatically stored in a csv in experiments/results. The recommended points, which are needed for inference regret, are all evaluated _after_ the full run is finished. These queries are appended to the same csv.



## Available options
Thanks to Wang & Jegelka (2017), this repository comes equipped with the following acquisition functions. MES-G (shortened to just MES) and EI were used in this paper to benchmark against.
1. Max-value Entropy Search with Gumbel sampling (MES-G) by Wang & Jegelka, 2017;
2. Max-value Entropy Search with random features (MES-R) by Wang & Jegelka, 2017;
3. Optimization as estimation (EST) by Wang et al., 2016. 
4. Gaussian process upper confidence bound (GP-UCB) by Auer, 2002; Srinivas et al., 2010;
5. Probability of improvement (PI) by Kushner, 1964;
6. Expected improvement (EI) by Mockus, 1974


The repository builds on [predictive entropy search](https://bitbucket.org/jmh233/codepesnips2014) (Hernández-Lobato et al., 2014).

## References
* --Anonymous author(s)--. Joint Entropy Search for Maximally-Informed Bayesian Optimization. Under review.
* Wang, Zi and Jegelka, Stefanie. Max-value Entropy Search for Efficient Bayesian Optimization. International Conference on Machine Learning (ICML), 2017.
* Auer, Peter. Using confidence bounds for exploitationexploration tradeoffs. Journal of Machine Learning Research, 3:397–422, 2002.
* Srinivas, Niranjan, Krause, Andreas, Kakade, Sham M, and Seeger, Matthias. Gaussian process optimization in the bandit setting: No regret and experimental design. In International Conference on Machine Learning (ICML), 2010.
* Kushner, Harold J. A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. Journal of Fluids Engineering, 86(1):97–106, 1964.
* Mockus, J. On Bayesian methods for seeking the extremum. In Optimization Techniques IFIP Technical Conference, 1974.
* Wang, Zi, Zhou, Bolei, and Jegelka, Stefanie. Optimization as estimation with Gaussian processes in bandit settings. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2016.
* Catto, Erin. Box2d, a 2D physics engine for games. http://box2d.org, 2011.
* Pybox2d, 2D Game Physics for Python. http://github.com/pybox2d/pybox2d.
* Hernández-Lobato, José Miguel, Hoffman, Matthew W, and Ghahramani, Zoubin. Predictive entropy search for efficient global optimization of black-box functions. In Advances in Neural Information Processing Systems (NIPS), 2014. https://bitbucket.org/jmh233/codepesnips2014
* Hennig, Philipp and Schuler, Christian J. Entropy search for information-efficient global optimization. Journal of Machine Learning Research, 13:1809–1837, 2012. http://www.probabilistic-optimization.org/Global.html
* Jarno Vanhatalo, Jaakko Riihimäki, Jouni Hartikainen, Pasi Jylänki, Ville Tolvanen, Aki Vehtari. GPstuff: Bayesian Modeling with Gaussian Processes. Journal of Machine Learning Research, 14(Apr):1175-1179, 2013.
* Kandasamy, Kirthevasan, Schneider, Jeff, and Poczos, Barnabas. High dimensional Bayesian optimisation and bandits via additive models. In International Conference on Machine Learning (ICML), 2015.
* Wang, Zi, Li, Chengtao, Jegelka, Stefanie, and Kohli, Pushmeet. Batched High-dimensional Bayesian Optimization via Structural Kernel Learning. International Conference on Machine Learning (ICML), 2017.
* Westervelt, Eric R, Grizzle, Jessy W, Chevallereau, Christine, Choi, Jun Ho, and Morris, Benjamin. Feedback control of dynamic bipedal robot locomotion, volume 28. CRC press, 2007.
