# geotensor

Geometric Low-Rank Tensor Completion for Color Image Inpainting.

## Motivation

- **Color image inpainting**: various missing patterns.

<table>
  <tr>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_mar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_mar.jpg?size=150" width="150px;" alt="Missing at random (MAR)"/><br /><sub><b>Missing at random (MAR)</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_rmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_rmar.jpg?size=150" width="150px;" alt="Row-wise MAR"/><br /><sub><b>Row-wise MAR</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_cmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_cmar.jpg?size=150" width="150px;" alt="Column-wise MAR"/><br /><sub><b>Column-wise MAR</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/xinychen/geotensor/blob/master/data/lena_rcmar.jpg"><img src="https://github.com/xinychen/geotensor/blob/master/data/lena_rcmar.jpg?size=150" width="150px;" alt="(Row, column)-wise MAR"/><br /><sub><b>(Row, column)-wise MAR</b></sub></a><br /></td>
  </tr>
</table>

- **Low-rank tensor completion**: smoothness modeling.

## Implementation

- Proposed Models

  - [LGTC (Nuclear Norm, NN)](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/GLTC-NN.ipynb)
  - [LGTC (Capped-L1)](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/GLTC-Capped-L1.ipynb)
  - [LGTC (LSP)](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/GLTC-LSP.ipynb)

- Competing Models

  - [HaLRTC](xxxx)
  - [None](xxxx)

## Reference

- **General Tensor Completion**

| No | Title | Year | PDF | Code |
|:--|:------|:----:|:---:|-----:|
|  1 | Tensor Completion for Estimating Missing Values in Visual Data | 2013 | [TPAMI](https://doi.org/10.1109/TPAMI.2012.39) | - |
|  2 | Efficient tensor completion for color image and video recovery: Low-rank tensor train | 2016 | [arxiv](https://arxiv.org/pdf/1606.01500.pdf) | - |
|  3 | Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks | 2017 | [NeurIPS](https://arxiv.org/abs/1704.06803)| [Python](https://github.com/fmonti/mgcnn) |
|  4 | Spatio-Temporal Signal Recovery Based on Low Rank and Differential Smoothness | 2018 | [IEEE](https://doi.org/10.1109/TSP.2018.2875886) | - |

- **Matrix/Tensor Completion + Nonconvex Regularization**

| No | Title | Year | PDF | Code |
|:--|:------|:----:|:---:|-----:|
| 1 | Generalized Singular Value Thresholding | 2015 | [AAAI](https://arxiv.org/abs/1412.2231) | - |
| 2 | Scalable Tensor Completion with Nonconvex Regularization | 2018 | [arxiv](http://arxiv.org/pdf/1807.08725v1.pdf) | - |
| 3 | Large-Scale Low-Rank Matrix Learning with Nonconvex Regularizers | 2018 | [TPAMI](https://ieeexplore.ieee.org/document/8416722/) | - |
| 4 | Nonconvex Robust Low-rank Matrix Recovery | 2018 | [arxiv](https://arxiv.org/pdf/1809.09237.pdf) | [Matlab](https://github.com/lixiao0982/Nonconvex-Robust-Low-rank-Matrix-Recovery) |
| 5 | Matrix Completion via Nonconvex Regularization: Convergence of the Proximal Gradient Algorithm | 2019 | [arxiv](http://arxiv.org/pdf/1903.00702v1.pdf) | [Matlab](https://github.com/FWen/nmc) |
| 6 | Efficient Nonconvex Regularized Tensor Completion with Structure-aware Proximal Iterations | 2019 | [ICML](http://proceedings.mlr.press/v97/yao19a/yao19a.pdf) | [Matlab](https://github.com/quanmingyao/FasTer) |
| 7 | Guaranteed Matrix Completion under Multiple Linear Transformations | 2019 | [CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Guaranteed_Matrix_Completion_Under_Multiple_Linear_Transformations_CVPR_2019_paper.pdf) | - |

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
| 10 | Generalized Singular Value Thresholding | 2014 | [AAAI](https://arxiv.org/pdf/1412.2231.pdf) | - |
