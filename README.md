# geotensor
Geometric Low-Rank Tensor Completion for Color Image Inpainting

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

- Low-rank tensor completion: graph modeling.

## Implementation

- [LGTC](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/GLTC.ipynb)
- [LGTC+](https://nbviewer.jupyter.org/github/xinychen/geotensor/blob/master/GLTC%20.ipynb)


## Reference

- **Matrix/Tensor Completion + Nonconvex Regularization**

| No | Title | Year | PDF | Code |
|:---|:------|:----:|:---:|-----:|
| 1  | Scalable Tensor Completion with Nonconvex Regularization | 2018 | [arxiv](http://arxiv.org/pdf/1807.08725v1.pdf) | - |
| 2  | Matrix Completion via Nonconvex Regularization: Convergence of the Proximal Gradient Algorithm | 2019 | [arxiv](http://arxiv.org/pdf/1903.00702v1.pdf) | [Matlab](https://github.com/FWen/nmc) |
| 3 | Efficient Nonconvex Regularized Tensor Completion with Structure-aware Proximal Iterations | 2019 | [ICML](http://proceedings.mlr.press/v97/yao19a/yao19a.pdf) | [Matlab](https://github.com/quanmingyao/FasTer) |
| 4 | Guaranteed Matrix Completion under Multiple Linear Transformations | 2019 | [CVPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Guaranteed_Matrix_Completion_Under_Multiple_Linear_Transformations_CVPR_2019_paper.pdf) | - |
| 5 | Large-Scale Low-Rank Matrix Learning with Nonconvex Regularizers | 2018 | [TPAMI](https://ieeexplore.ieee.org/document/8416722/) | - |
| 6 | Nonconvex Robust Low-rank Matrix Recovery | 2018 | [arxiv](https://arxiv.org/pdf/1809.09237.pdf) | [Matlab](https://github.com/lixiao0982/Nonconvex-Robust-Low-rank-Matrix-Recovery) |

- **Rank Approximation + Nonconvex Regularization**

| No | Title | Year | PDF | Code |
|:---|:------|:----:|:---:|-----:|
| 1  | Minimizing the Maximal Rank | 2016 | [CVPR](http://www1.maths.lth.se/matematiklth/vision/publdb/reports/pdf/bylow-olsson-etal-cvpr-16.pdf) | - |
| 2 | Non-Convex Relaxations for Rank Regularization | 2019 | [slide](https://icerm.brown.edu/materials/Slides/sp-s19-w3/Non-Convex_Relaxations_for_Rank_Regularization_]_Carl_Olsson,_Chalmers_University_of_Technology_and_Lund_University.pdf) | - |
| 3 | Geometry and Regularization in Nonconvex Low-Rank Estimation | 2019 | [slide](http://users.ece.cmu.edu/~yuejiec/papers/NonconvexLowrank.pdf) | - |
| 4 | Inexact Proximal Gradient Methods for Non-Convex and Non-Smooth Optimization | 2018 | [AAAI](http://www.pitt.edu/~zhh39/others/aaaigu18a.pdf) | - |
| 5 | Non-Convex Rank/Sparsity Regularization and Local Minima | 2017 | [ICCV](http://openaccess.thecvf.com/content_ICCV_2017/papers/Olsson_Non-Convex_RankSparsity_Regularization_ICCV_2017_paper.pdf), [Supp](http://openaccess.thecvf.com/content_ICCV_2017/supplemental/Olsson_Non-Convex_RankSparsity_Regularization_ICCV_2017_supplemental.pdf) | - |
| 6 | Convex Low Rank Approximation | 2016 | [IJCV](http://www.maths.lth.se/vision/publdb/reports/pdf/larsson-olsson-ijcv-16.pdf) | - |
