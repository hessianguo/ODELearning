# Derivative estimation by RKHS regularization for learning dynamics from time-series data
Learning the governing equations from time-series data has gained increasing attention due to its potential to extract useful dynamics from real-world data. Despite significant progress, it becomes challenging when when the data is noisy, especially when derivatives need to be calculated. To reduce the effect of noise, we propose a method that can fit the derivative and trajectory simultaneously from noisy time-series data. Our approach treats derivative estimation as an inverse problem involving integral operators in the forward model and estimates the derivative function by solving a regularization problem in a vector-valued reproducing kernel Hilbert space (vRKHS). We establish an integral-form representer theorem that allows us to compute the regularized solution by solving a finite-dimensional problem and efficiently estimate the optimal regularization parameter. By embedding the dynamics in a vRKHS and using the fitted derivative and trajectory, we can recover the dynamics from noisy data by only solving a linear regularization problem. Several numerical experiments are performed to demonstrate the effectiveness and efficiency of our method.

It is the implementation of the manuscript 	arXiv:2504.01289 [math.NA]. 

## References

[1] Hailong Guo and Haibo Li. *Derivative estimation by RKHS regularization for learning dynamics from time-series data*. arXiv:2504.01289 [math.NA], 2025.  
📄 [Read on arXiv](https://arxiv.org/abs/2504.01289)

## BibTeX

```bibtex
@misc{guo2025derivativeestimationrkhsregularization,
  title          = {Derivative estimation by RKHS regularization for learning dynamics from time-series data}, 
  author         = {Hailong Guo and Haibo Li},
  year           = {2025},
  eprint         = {2504.01289},
  archivePrefix  = {arXiv},
  primaryClass   = {math.NA},
  url            = {https://arxiv.org/abs/2504.01289}, 
}
