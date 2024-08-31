# Awesome-Federated-Transfer-Learning
Federated Transfer Learning

# Table of Contents
* [Federated Transfer Learning challenges](#federated-transfer-learning-challenges)
  * [DHFTL](#dhftl)
    * [System](#system)
    * [Incremental](#incremental)
  * [MAFTL](#maftl)
  * [SSFTL](#ssftl)
  * [USFTL](#usftl)
  * [HOFTL](#hoftl)
  * [HEFTL](#heftl)

## Federated Transfer Learning challenges
### DHFTL  
##### System
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |
 | [Li T, Sahu A K, Zaheer M, Sanjabi M, Talwalkar A, Smith V. Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2020, 2: 429–450](https://arxiv.org/abs/1812.06127) | Parameter Restriction | MLSys | 2020 | [Pytorch](https://github.com/litian96/FedProx) |
 | [Shin J, Li Y, Liu Y, Lee S J. Fedbalancer: data and pace control for efficient federated learning on heterogeneous clients. In: Proceedings of the 20th Annual International Conference on Mobile Systems, Applications and Services. 2022, 436–449](https://arxiv.org/abs/2201.01601) | Instance Selection | MobiSys | 2022 | [Pytorch](https://github.com/jaemin-shin/fedbalancer) |
 | [Pilla L L. Optimal task assignment for heterogeneous federated learning devices. In: 2021 IEEE International Parallel and Distributed Processing Symposium (IPDPS). 2021, 661–670](https://ieeexplore.ieee.org/document/9460535) | Instance Selection | IEEE | 2021 | [Python/Torch](https://github.com/llpilla/olar-federated-learning) | 
 | [Li A, Sun J, Li P, Pu Y, Li H, Chen Y. Hermes: an efficient federated learning framework for heterogeneous mobile clients. In: Proceedings of the 27th Annual International Conference on Mobile Computing and Networking. 2021, 420–437](https://dl.acm.org/doi/10.1145/3447993.3483278) | Parameter Decoupling | ACM MobiCom | 2021 | N/A |
 | [Chai Z, Chen Y, Anwar A, Zhao L, Cheng Y, Rangwala H. Fedat: A high-performance and communication-efficient federated learning system with asynchronous tiers. In: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2021, 1–16](https://arxiv.org/abs/2010.05958) | Parameter Decoupling, Model Clustering | SC'21 | 2021 | N/A |
 | [Li A, Sun J, Zeng X, Zhang M, Li H, Chen Y. Fedmask: Joint computation and communication efficient personalized federated learning via heterogeneous masking. In: Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems. 2021, 42–55](https://dl.acm.org/doi/10.1145/3485730.3485929) | Parameter Pruning | SenSys | 2021 | N/A |
 | [Yang Z, Sun Q. Personalized heterogeneity-aware federated search towards better accuracy and energy efficiency. In: Proceedings of the 41st IEEE/ACM International Conference on Computer-Aided Design. 2022, 1–9](https://dl.acm.org/doi/abs/10.1145/3508352.3549403) | Parameter Pruning,  Model Clustering | IEEE/ACM | 2022 | N/A |
 | [Ilhan F, Su G, Liu L. Scalefl: Resource-adaptive federated learning with heterogeneous clients. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023, 24532–24541](https://paperswithcode.com/paper/scalefl-resource-adaptive-federated-learning) | Parameter Pruning, Knowledge Distillation | CVPR | 2023 | [Pytorch](https://github.com/git-disl/scale-fl) |
 | [McMahan B, Moore E, Ramage D, Hampson S, Arcas y B A. Communication-efficient learning of deep networks from decentralized data. In: Artificial intelligence and statistics. 2017, 1273–1282](https://arxiv.org/abs/1602.05629) | Model Weighting | AISTATS | 2017 | N/A |
 | [Deng Y, Lyu F, Ren J, Wu H, Zhou Y, Zhang Y, Shen X. Auction: Automated and quality-aware client selection framework for efficient federated learning. IEEE Transactions on Parallel and Distributed Systems, 2021, 33(8): 1996–2009](https://ieeexplore.ieee.org/document/9647925) | Model Selection | IEEE | 2021 | N/A |
 | [Chai Z, Ali A, Zawad S, Truex S, Anwar A, Baracaldo N, Zhou Y, Ludwig H, Yan F, Cheng Y. Tifl: A tier-based federated learning system. In: Proceedings of the 29th international symposium on high-performance parallel and distributed computing. 2020, 125–136](https://arxiv.org/abs/2001.09249) | Model Selection | HPDC | 2020 | N/A |
 | [Su L, Zhou R, Wang N, Fang G, Li Z. An online learning approach for client selection in federated edge learning under budget constraint. In: Proceedings of the 51st International Conference on Parallel Processing. 2022, 1–11](https://dl.acm.org/doi/abs/10.1145/3545008.3545062) | Model Selection | ICPP | 2022 | N/A |
 | [Yoon J, Park G, Jeong W, Hwang S J. Bitwidth heterogeneous federated learning with progressive weight dequantization. In: International Conference on Machine Learning. 2022, 25552–25565](https://arxiv.org/abs/2202.11453) | Model Selection | ICML | 2022 | N/A |
 | [Yang M, Wang X, Zhu H, Wang H, Qian H. Federated learning with class imbalance reduction. In: 2021 29th European Signal Processing Conference (EUSIPCO). 2021, 2174–2178](https://arxiv.org/abs/2011.11266) | Model Selection, Model Weighting | IEEE | 2021 | N/A |
 | [Qu Z, Duan R, Chen L, Xu J, Lu Z, Liu Y. Context-aware online client selection for hierarchical federated learning. IEEE Transactions on Parallel and Distributed Systems, 2022, 33(12): 4353–4367](https://arxiv.org/abs/2112.00925) | Model Selection, Model Clustering | IEEE | 2022 | N/A |
 | [Li G, Hu Y, Zhang M, Liu J, Yin Q, Peng Y, Dou D. Fedhisyn: A hierarchical synchronous federated learning framework for resource and data heterogeneity. In: Proceedings of the 51st International Conference on Parallel Processing. 2022, 1–11](https://arxiv.org/abs/2206.10546) | Model Clustering | ICPP | 2022 | N/A |
 | [Zhang L, Wu D, Yuan X. Fedzkt: Zero-shot knowledge transfer towards resource-constrained federated learning with heterogeneous on-device models. In: 2022 IEEE 42nd International Conference on Distributed Computing Systems (ICDCS). 2022, 928–938](https://arxiv.org/abs/2109.03775) | Knowledge Distillation | IEEE | 2022 | N/A |
 | [Itahara S, Nishio T, Koda Y, Morikura M, Yamamoto K. Distillation-based semi-supervised federated learning for communication-efficient collaborative training with non-iid private data. IEEE Transactions on Mobile Computing, 2021, 22(1): 191–205](https://arxiv.org/abs/2008.06180) | Knowledge Distillation | IEEE | 2021 | N/A |
##### Incremental
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |
 | [Dong J, Wang L, Fang Z, Sun G, Xu S, Wang X, Zhu Q. Federated class-incremental learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, 10164–10173](https://arxiv.org/abs/2203.11473) | Consistency Regularization, Model Selection; | CVPR | 2022 | [Pytorch](https://github.com/conditionwang/fcil) |
 | [Yoon J, Jeong W, Lee G, Yang E, Hwang S J. Federated continual learning with weighted inter-client transfer. In: International Conference on Machine Learning. 2021, 12073–12086](https://arxiv.org/abs/2003.03196) | Parameter Decoupling, Model Interpolation | ICML | 2021 | [Tensorflow](https://github.com/wyjeong/FedWeIT) |
 | [Su L, Zhou R, Wang N, Fang G, Li Z. An online learning approach for client selection in federated edge learning under budget constraint. In: Proceedings of the 51st International Conference on Parallel Processing. 2022, 1–11](https://dl.acm.org/doi/10.1145/3545008.3545062) | Model Selection | ICPP | 2022 | N/A |

### MAFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

### SSFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

### USFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

### HOFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

### HEFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

 
