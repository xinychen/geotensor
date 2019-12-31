# geotensor

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
[![repo size](https://img.shields.io/github/repo-size/xinychen/geotensor.svg)](https://github.com/xinychen/geotensor/archive/master.zip)
[![GitHub stars](https://img.shields.io/github/stars/xinychen/geotensor.svg?logo=github&label=Stars&logoColor=white)](https://github.com/xinychen/geotensor)


## Motivation

- **Color image inpainting**: Various missing patterns.

<table>
  <tr>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_mar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_mar.jpg?size=150" width="150px;" alt="Missing at random (MAR)"/><br /><sub><b>Missing at random (MAR)</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_rmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_rmar.jpg?size=150" width="150px;" alt="Row-wise MAR"/><br /><sub><b>Row-wise MAR</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_cmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_cmar.jpg?size=150" width="150px;" alt="Column-wise MAR"/><br /><sub><b>Column-wise MAR</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_rcmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_rcmar.jpg?size=150" width="150px;" alt="(Row, column)-wise MAR"/><br /><sub><b>(Row, column)-wise MAR</b></sub></a><br /></td>
  </tr>
</table>

- **Low-rank tensor completion**: Characterizing images with graph (e.g., adjacent smoothness matrix based graph regularizer).

## Implementation

- Proposed Models

  - [GLTC-NN (Nuclear Norm)](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/GLTC-NN.ipynb)
  - [GLTC-Geman (nonconvex)](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/GLTC-Geman.ipynb)
  - [GTC (without low-rank assumption)](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/GTC.ipynb)


One notable thing is that unlike the complex equations in our models, our Python implementation (relies on `numpy`) is extremely easy to work with. Take **`GLTC-Geman`** as an example, its kernel only has few lines:

```python
def supergradient(s_hat, lambda0, theta):
    """Supergradient of the Geman function."""
    return (lambda0 * theta / (s_hat + theta) ** 2)

def GLTC_Geman(dense_tensor, sparse_tensor, alpha, beta, rho, theta, maxiter):
    """Main function of the GLTC-Geman."""
    dim0 = sparse_tensor.ndim
    dim1, dim2, dim3 = sparse_tensor.shape
    dim = np.array([dim1, dim2, dim3])
    binary_tensor = np.zeros((dim1, dim2, dim3))
    binary_tensor[np.where(sparse_tensor != 0)] = 1
    tensor_hat = sparse_tensor.copy()
    
    X = np.zeros((dim1, dim2, dim3, dim0)) # \boldsymbol{\mathcal{X}} (n1*n2*3*d)
    Z = np.zeros((dim1, dim2, dim3, dim0)) # \boldsymbol{\mathcal{Z}} (n1*n2*3*d)
    T = np.zeros((dim1, dim2, dim3, dim0)) # \boldsymbol{\mathcal{T}} (n1*n2*3*d)
    for k in range(dim0):
        X[:, :, :, k] = tensor_hat
        Z[:, :, :, k] = tensor_hat
    
    D1 = np.zeros((dim1 - 1, dim1)) # (n1-1)-by-n1 adjacent smoothness matrix
    for i in range(dim1 - 1):
        D1[i, i] = -1
        D1[i, i + 1] = 1
    D2 = np.zeros((dim2 - 1, dim2)) # (n2-1)-by-n2 adjacent smoothness matrix
    for i in range(dim2 - 1):
        D2[i, i] = -1
        D2[i, i + 1] = 1
        
    w = []
    for k in range(dim0):
        u, s, v = np.linalg.svd(ten2mat(Z[:, :, :, k], k), full_matrices = 0)
        w.append(np.zeros(len(s)))
        for i in range(len(np.where(s > 0)[0])):
            w[k][i] = supergradient(s[i], alpha, theta)

    for iters in range(maxiter):
        for k in range(dim0):
            u, s, v = np.linalg.svd(ten2mat(X[:, :, :, k] + T[:, :, :, k] / rho, k), full_matrices = 0)
            for i in range(len(np.where(w[k] > 0)[0])):
                s[i] = max(s[i] - w[k][i] / rho, 0)
            Z[:, :, :, k] = mat2ten(np.matmul(np.matmul(u, np.diag(s)), v), dim, k)
            var = ten2mat(rho * Z[:, :, :, k] - T[:, :, :, k], k)
            if k == 0:
                var0 = mat2ten(np.matmul(inv(beta * np.matmul(D1.T, D1) + rho * np.eye(dim1)), var), dim, k)
            elif k == 1:
                var0 = mat2ten(np.matmul(inv(beta * np.matmul(D2.T, D2) + rho * np.eye(dim2)), var), dim, k)
            else:
                var0 = Z[:, :, :, k] - T[:, :, :, k] / rho
            X[:, :, :, k] = np.multiply(1 - binary_tensor, var0) + sparse_tensor
            
            uz, sz, vz = np.linalg.svd(ten2mat(Z[:, :, :, k], k), full_matrices = 0)
            for i in range(len(np.where(sz > 0)[0])):
                w[k][i] = supergradient(sz[i], alpha, theta)
        tensor_hat = np.mean(X, axis = 3)
        for k in range(dim0):
            T[:, :, :, k] = T[:, :, :, k] + rho * (X[:, :, :, k] - Z[:, :, :, k])
            X[:, :, :, k] = tensor_hat.copy()

    return tensor_hat
```

> Have fun if you work with our code!

- Competing Models

  - [Tmac-TT (Tensor Train)](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/Tmac-TT.ipynb)
  - [HaLRTC](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/HaLRTC.ipynb)
  - [Image Spatial Filtering](https://github.com/xinychen/geotensor/blob/master/Image_recovery_filtering.ipynb)

- Inpainting Examples (by **`GLTC-Geman`**)

<table>
  <tr>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_mar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_mar.jpg?size=150" width="150px;" alt="Missing at random (MAR)"/><br /><sub><b>Missing at random (MAR)</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_rmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_rmar.jpg?size=150" width="150px;" alt="Row-wise MAR"/><br /><sub><b>Row-wise MAR</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_cmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_cmar.jpg?size=150" width="150px;" alt="Column-wise MAR"/><br /><sub><b>Column-wise MAR</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_rcmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_rcmar.jpg?size=150" width="150px;" alt="(Row, column)-wise MAR"/><br /><sub><b>(Row, column)-wise MAR</b></sub></a><br /></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/GLTC_Geman_lena_mar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/GLTC_Geman_lena_mar.jpg?size=150" width="150px;" alt="RSE = 6.74%"/><br /><sub><b>RSE = 6.74%</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/GLTC_Geman_lena_rmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/GLTC_Geman_lena_rmar.jpg?size=150" width="150px;" alt="RSE = 8.20%"/><br /><sub><b>RSE = 8.20%</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/GLTC_Geman_lena_cmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/GLTC_Geman_lena_cmar.jpg?size=150" width="150px;" alt="RSE = 10.80%"/><br /><sub><b>RSE = 10.80%</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/GLTC_Geman_lena_rcmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/GLTC_Geman_lena_rcmar.jpg?size=150" width="150px;" alt="RSE = 8.38%"/><br /><sub><b>RSE = 8.38%</b></sub></a><br /></td>
  </tr>
</table>


## Reference

- **General Matrix/Tensor Completion**

| No | Title | Year | PDF | Code |
|:---|:------|:----:|:---:|-----:|
|  1 | Tensor Completion for Estimating Missing Values in Visual Data | 2013 | [TPAMI](https://doi.org/10.1109/TPAMI.2012.39) | - |
|  2 | Efficient tensor completion for color image and video recovery: Low-rank tensor train | 2016 | [arxiv](https://arxiv.org/pdf/1606.01500.pdf) | - |
|  3 | Tensor Robust Principal Component Analysis: Exact Recovery of Corrupted Low-Rank Tensors via Convex Optimization | 2016 | [CVPR](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lu_Tensor_Robust_Principal_CVPR_2016_paper.pdf) | [Matlab](https://github.com/canyilu/Tensor-Robust-Principal-Component-Analysis-TRPCA) |
|  4 | Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks | 2017 | [NeurIPS](https://arxiv.org/abs/1704.06803)| [Python](https://github.com/fmonti/mgcnn) |
|  5 | Efficient Low Rank Tensor Ring Completion | 2017 | [ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Efficient_Low_Rank_ICCV_2017_paper.pdf) | [Matlab](https://github.com/wangwenqi1990/TensorRingCompletion) |
|  6 | Spatio-Temporal Signal Recovery Based on Low Rank and Differential Smoothness | 2018 | [IEEE](https://doi.org/10.1109/TSP.2018.2875886) | - |
|  7 | Exact Low Tubal Rank Tensor Recovery from Gaussian Measurements | 2018 | [IJCAI](https://canyilu.github.io/publications/2018-IJCAI-Atomic.pdf) | [Matlab](https://github.com/canyilu/tensor-completion-tensor-recovery) |
|  8 | Tensor Robust Principal Component Analysis with A New Tensor Nuclear Norm | 2018 | [TPAMI](https://arxiv.org/abs/1804.03728) | [Matlab](https://github.com/canyilu/Tensor-Robust-Principal-Component-Analysis-TRPCA) |

- **Singular Value Thresholding (SVT)**

| No | Title | Year | PDF | Code |
|:---|:------|:----:|:---:|-----:|
| 1 | A General Iterative Shrinkage and Thresholding Algorithm for Non-convex Regularized Optimization Problems | 2013 | [ICML](http://proceedings.mlr.press/v28/gong13a.pdf) | - |
| 2 | Fast Randomized Singular Value Thresholding for Nuclear Norm Minimization | 2015 | [CVPR](http://openaccess.thecvf.com/content_cvpr_2015/papers/Oh_Fast_Randomized_Singular_2015_CVPR_paper.pdf) | - |
| 3 | A Fast Implementation of Singular Value Thresholding Algorithm using Recycling Rank Revealing Randomized Singular Value Decomposition | 2017 | [arxiv](https://arxiv.org/pdf/1704.05528.pdf) | - |
| 4 | Fast Randomized Singular Value Thresholding for Low-rank Optimization | 2018 | [TPAMI](https://arxiv.org/pdf/1509.00296v2.pdf) | - |
| 5 | Fast Parallel Randomized QR with Column Pivoting Algorithms for Reliable Low-rank Matrix Approximations | 2018 | [arxiv](https://arxiv.org/pdf/1804.05138.pdf) | - |
| 6 | Low-Rank Matrix Approximations with Flip-Flop Spectrum-Revealing QR Factorization | 2018 | [arxiv](https://arxiv.org/pdf/1803.01982.pdf) | - |

- **Proximal Methods**

| No | Title | Year | PDF | Code |
|:---|:------|:----:|:---:|-----:|
|  1 | Accelerated Proximal Gradient Methods for Nonconvex Programming | 2015 | [NIPS](https://papers.nips.cc/paper/5728-accelerated-proximal-gradient-methods-for-nonconvex-programming.pdf) | [Supp](https://papers.nips.cc/paper/5728-accelerated-proximal-gradient-methods-for-nonconvex-programming-supplemental.zip) |
|  2 | Incorporating Nesterov Momentum into Adam | 2016 | [ICLR](https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ) | - |

- **Fast Alternating Direction Method of Multipliers**

| No | Title | Year | PDF | Code |
|:---|:------|:----:|:---:|-----:|
|  1 | Differentiable Linearized ADMM | 2019 | [ICML](http://proceedings.mlr.press/v97/xie19c/xie19c.pdf) | - |
|  2 | Faster Stochastic Alternating Direction Method of Multipliers for Nonconvex Optimization | 2019 | [ICML](http://proceedings.mlr.press/v97/huang19a/huang19a.pdf) | - |


- **Tensor Train Decomposition**

| No | Title | Year | PDF | Code |
|:---|:------|:----:|:---:|-----:|
|  1 | Math Lecture 671: Tensor Train decomposition methods | 2016 | [slide](http://www-personal.umich.edu/~coronae/Talk_UM_TT_lecture1.pdf) | - |
|  2 | Introduction to the Tensor Train Decomposition and Its Applications in Machine Learning | 2016 | [slide](https://bayesgroup.github.io/team/arodomanov/tt_hse16_slides.pdf) | - |

- **Matrix/Tensor Completion + Nonconvex Regularization**

| No | Title | Year | PDF | Code |
|:--|:------|:----:|:---:|-----:|
| 1 | Fast and Accurate Matrix Completion via Truncated Nuclear Norm Regularization | 2013 | [TPAMI](https://ieeexplore.ieee.org/document/6389682/) | - |
| 2 | Generalized  noncon-vex nonsmooth low-rank minimization | 2014 | [CVPR](https://doi.org/10.1109/CVPR.2014.526) | [Matlab](https://github.com/canyilu/IRNN) |
| 3 | Generalized Singular Value Thresholding | 2015 | [AAAI](https://arxiv.org/abs/1412.2231) | - |
| 4 | Partial Sum Minimization of Singular Values in Robust PCA: Algorithm and Applications | 2016 | [TPAMI](https://arxiv.org/pdf/1503.01444.pdf) | - |
| 5 | Efficient Inexact Proximal Gradient Algorithm for Nonconvex Problems | 2016 | [arxiv](https://arxiv.org/pdf/1612.09069.pdf) | - |
| 6 | Scalable Tensor Completion with Nonconvex Regularization | 2018 | [arxiv](http://arxiv.org/pdf/1807.08725v1.pdf) | - |
| 7 | Large-Scale Low-Rank Matrix Learning with Nonconvex Regularizers | 2018 | [TPAMI](https://ieeexplore.ieee.org/document/8416722/) | - |
| 8 | Nonconvex Robust Low-rank Matrix Recovery | 2018 | [arxiv](https://arxiv.org/pdf/1809.09237.pdf) | [Matlab](https://github.com/lixiao0982/Nonconvex-Robust-Low-rank-Matrix-Recovery) |
| 9 | Matrix Completion via Nonconvex Regularization: Convergence of the Proximal Gradient Algorithm | 2019 | [arxiv](http://arxiv.org/pdf/1903.00702v1.pdf) | [Matlab](https://github.com/FWen/nmc) |
| 10 | Efficient Nonconvex Regularized Tensor Completion with Structure-aware Proximal Iterations | 2019 | [ICML](http://proceedings.mlr.press/v97/yao19a/yao19a.pdf) | [Matlab](https://github.com/quanmingyao/FasTer) |
| 11 | Guaranteed Matrix Completion under Multiple Linear Transformations | 2019 | [CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Guaranteed_Matrix_Completion_Under_Multiple_Linear_Transformations_CVPR_2019_paper.pdf) | - |

- **Rank Approximation + Nonconvex Regularization**

| No | Title | Year | PDF | Code |
|:--|:------|:----:|:---:|-----:|
|  1 | A General Iterative Shrinkage and Thresholding Algorithm for Non-convex Regularized Optimization Problems | 2013 | [ICML](http://proceedings.mlr.press/v28/gong13a.pdf) | - |
|  2 | Rank Minimization with Structured Data Patterns | 2014 | [ECCV](http://www1.maths.lth.se/matematiklth/vision/publdb/reports/pdf/larsson-olsson-etal-eccv-14.pdf) | - |
|  3 | Minimizing the Maximal Rank | 2016 | [CVPR](http://www1.maths.lth.se/matematiklth/vision/publdb/reports/pdf/bylow-olsson-etal-cvpr-16.pdf) | - |
|  4 | Convex Low Rank Approximation | 2016 | [IJCV](http://www.maths.lth.se/vision/publdb/reports/pdf/larsson-olsson-ijcv-16.pdf) | - |
|  5 | Non-Convex Rank/Sparsity Regularization and Local Minima | 2017 | [ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Olsson_Non-Convex_RankSparsity_Regularization_ICCV_2017_paper.pdf), [Supp](http://openaccess.thecvf.com/content_ICCV_2017/supplemental/Olsson_Non-Convex_RankSparsity_Regularization_ICCV_2017_supplemental.pdf) | - |
|  6 | A Non-Convex Relaxation for Fixed-Rank Approximation | 2017 | [ICCV](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w25/Olsson_A_Non-Convex_Relaxation_ICCV_2017_paper.pdf) | - |
|  7 | Inexact Proximal Gradient Methods for Non-Convex and Non-Smooth Optimization | 2018 | [AAAI](http://www.pitt.edu/~zhh39/others/aaaigu18a.pdf) | - |
|  8 | Non-Convex Relaxations for Rank Regularization | 2019 | [slide](https://icerm.brown.edu/materials/Slides/sp-s19-w3/Non-Convex_Relaxations_for_Rank_Regularization_]_Carl_Olsson,_Chalmers_University_of_Technology_and_Lund_University.pdf) | - |
|  9 | Geometry and Regularization in Nonconvex Low-Rank Estimation | 2019 | [slide](http://users.ece.cmu.edu/~yuejiec/papers/NonconvexLowrank.pdf) | - |
|  10 | Large-Scale Low-Rank Matrix Learning with Nonconvex Regularizers | 2018 | [IEEE TPAMI](https://arxiv.org/pdf/1708.00146.pdf) | - |

- **Weighted Nuclear Norm Minimization**

| No | Title | Year | PDF | Code |
|:--|:------|:----:|:---:|-----:|
| 1 | Weighted Nuclear Norm Minimization with Application to Image Denoising | 2014 | [CVPR](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Gu_Weighted_Nuclear_Norm_2014_CVPR_paper.pdf) | [Matlab](http://www4.comp.polyu.edu.hk/~cslzhang/code/WNNM_code.zip) |
| 2 | A Nonconvex Relaxation Approach for Rank Minimization Problems | 2015 | [AAAI](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2015/Xiaowei-Zhong-AAAI.pdf) | - |
| 3 | Multi-Scale Weighted Nuclear Norm Image Restoration | 2018 | [CVPR](https://www.zpascal.net/cvpr2018/Yair_Multi-Scale_Weighted_Nuclear_CVPR_2018_paper.pdf) | [Matlab](https://github.com/noamyairTC/MSWNNM) |
| 4 | On the Optimal Solution of Weighted Nuclear Norm Minimization | - | [PDF](http://www4.comp.polyu.edu.hk/~cslzhang/paper/WNNM_GS.pdf) | - |


Collaborators
--------------

<table>
  <tr>
    <td align="center"><a href="https://github.com/xinychen"><img src="https://github.com/xinychen.png?size=80" width="80px;" alt="Xinyu Chen"/><br /><sub><b>Xinyu Chen</b></sub></a><br /><a href="https://github.com/xinychen/geotensor/commits?author=xinychen" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Vadermit"><img src="https://github.com/Vadermit.png?size=80" width="80px;" alt="Jinming Yang"/><br /><sub><b>Jinming Yang</b></sub></a><br /><a href="https://github.com/xinychen/geotensor/commits?author=Vadermit" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/lijunsun"><img src="https://github.com/lijunsun.png?size=80" width="80px;" alt="Lijun Sun"/><br /><sub><b>Lijun Sun</b></sub></a><br /><a href="https://github.com/xinychen/geotensor/commits?author=lijunsun" title="Code">💻</a></td>
  </tr>
</table>

See the list of [contributors](https://github.com/xinychen/geotensor/graphs/contributors) who participated in this project.


License
--------------

This work is released under the MIT license.

