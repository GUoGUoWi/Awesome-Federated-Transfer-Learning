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
 | [Qu Z, Duan R, Chen L, Xu J, Lu Z, Liu Y. Context-aware online client selection for hierarchical federated learning. IEEE Transactions on Parallel and Distributed Systems, 2022, 33(12): 4353–4367](https://arxiv.org/abs/2112.00925) | Model Selection, Model Clustering | IEEE | 2022 | N/A |
 | [Li G, Hu Y, Zhang M, Liu J, Yin Q, Peng Y, Dou D. Fedhisyn: A hierarchical synchronous federated learning framework for resource and data heterogeneity. In: Proceedings of the 51st International Conference on Parallel Processing. 2022, 1–11](https://arxiv.org/abs/2206.10546) | Model Clustering | ICPP | 2022 | N/A |
 | [Zhang L, Wu D, Yuan X. Fedzkt: Zero-shot knowledge transfer towards resource-constrained federated learning with heterogeneous on-device models. In: 2022 IEEE 42nd International Conference on Distributed Computing Systems (ICDCS). 2022, 928–938](https://arxiv.org/abs/2109.03775) | Knowledge Distillation | IEEE | 2022 | N/A |
 | [Itahara S, Nishio T, Koda Y, Morikura M, Yamamoto K. Fedzkt: Zero-shot knowledge transfer towards resource-constrained federated learning with heterogeneous on-device models. IEEE Transactions on Mobile Computing, 2021, 22(1): 191–205](https://arxiv.org/abs/2008.06180) | Knowledge Distillation | IEEE | 2021 | N/A |
 | [Chen M, Yang Z, Saad W, Yin C, Poor H V, Cui S. A joint learning and communications framework for federated learning over wireless networks. IEEE Transactions on Wireless Communications, 2020, 20(1): 269– 283](https://ieeexplore.ieee.org/document/9210812) | Model Selection | IEEE | 2020 | [Matlab](https://github.com/mzchen0/Wireless-FL?tab=readme-ov-file) |
 | [Yang H H, Arafa A, Quek T Q, Poor H V. Age-based scheduling policy for federated learning in mobile edge networks. In: ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2020, 8743–8747](https://arxiv.org/abs/1910.14648) | Model Selection | ICASSP | 2020 | N/A |
 | [Nishio T, Yonetani R. Client selection for federated learning with heterogeneous resources in mobile edge. In: ICC 2019-2019 IEEE international conference on communications (ICC). 2019, 1–7](https://arxiv.org/abs/1804.08333) | Model Selection | IEEE | 2019 | N/A |
 |[Xia W, Quek T Q, Guo K, Wen W, Yang H H, Zhu H. Multi-armed bandit-based client scheduling for federated learning. IEEE Transactions on Wireless Communications, 2020, 19(11): 7108–7123](https://arxiv.org/abs/2007.02315) | Model Selection | IEEE | 2020 | N/A |
 | [Huang T, Lin W, Wu W, He L, Li K, Zomaya A Y. An efficiency-boosting client selection scheme for federated learning with fairness guarantee. IEEE Transactions on Parallel and Distributed Systems, 2020, 32(7): 1552–1564](https://arxiv.org/abs/2011.01783) | Model Selection | IEEE | 2020 | N/A |
 | [Li L, Duan M, Liu D, Zhang Y, Ren A, Chen X, Tan Y, Wang C. Fedsae: A novel self-adaptive federated learning framework in heterogeneous systems. In: 2021 International Joint Conference on Neural Networks (IJCNN). 2021, 1–10](https://ieeexplore.ieee.org/document/9533876) | Model Selection | IEEE | 2021 | N/A |
 | [Cox B, Chen L Y, Decouchant J. Aergia: leveraging heterogeneity in federated learning systems. In: Proceedings of the 23rd ACM/IFIP International Middleware Conference. 2022, 107–120](https://arxiv.org/abs/2210.06154) | Model Selection | ACM/IFIP | 2022 | [Python](https://github.com/bacox/fltk) |
 | [Li C, Zeng X, Zhang M, Cao Z. Pyramidfl: A fine-grained client selection framework for efficient federated learning. In: Proceedings of the 28th Annual International Conference on Mobile Computing And Networking. 2022, 158–171](https://dl.acm.org/doi/abs/10.1145/3495243.3517017) | Model Selection | MobiCom | 2022 | [Python](https://github.com/liecn/PyramidFL) | 
##### Incremental
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |
 | [Dong J, Wang L, Fang Z, Sun G, Xu S, Wang X, Zhu Q. Federated class-incremental learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022, 10164–10173](https://arxiv.org/abs/2203.11473) | Consistency Regularization, Model Selection; | CVPR | 2022 | [Pytorch](https://github.com/conditionwang/fcil) |
 | [Yoon J, Jeong W, Lee G, Yang E, Hwang S J. Federated continual learning with weighted inter-client transfer. In: International Conference on Machine Learning. 2021, 12073–12086](https://arxiv.org/abs/2003.03196) | Parameter Decoupling, Model Interpolation | ICML | 2021 | [Tensorflow](https://github.com/wyjeong/FedWeIT) |
 | [Su L, Zhou R, Wang N, Fang G, Li Z. An online learning approach for client selection in federated edge learning under budget constraint. In: Proceedings of the 51st International Conference on Parallel Processing. 2022, 1–11](https://dl.acm.org/doi/10.1145/3545008.3545062) | Model Selection | ICPP | 2022 | N/A |

### MAFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |
 | [Wu Z, Li Q, He B. Practical vertical federated learning with unsupervised representation learning. IEEE Transactions on Big Data, 2022](https://arxiv.org/abs/2208.10278) | Feature Augmentation | IEEE | 2022 | [Pytorch](https://github.com/jerrylife/fedonce) |
 | [Liao Y, Ma L, Zhou B, Zhao X, Xie F. Draftfed: A draft-based personalized federated learning approach for heterogeneous convolutional neural networks. IEEE Transactions on Mobile Computing, 2023](https://ieeexplore.ieee.org/document/10145872) | Feature Mapping | IEEE | 2023 | N/A |
 | [Makhija D, Han X, Ho N, Ghosh J. Architecture agnostic federated learning for neural networks. In: International Conference on Machine Learning. 2022, 14860–14870](https://arxiv.org/abs/2202.07757) | Feature Alignment | PMLR | 2022 | N/A |
 | [Liu Y, Guo S, Zhang J, Zhou Q, Wang Y, Zhao X. Feature correlation-guided knowledge transfer for federated self-supervised learning. arXiv preprint arXiv:2211.07364, 2022](https://arxiv.org/abs/2211.07364) | Feature Alignment | arXiv | 2022 | N/A |
 | [Tan Y, Long G, Liu L, Zhou T, Lu Q, Jiang J, Zhang C. Fedproto: Federated prototype learning across heterogeneous clients. In: Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 8432–8440](https://arxiv.org/abs/2105.00243) | Feature Alignment, Model Clustering | AAAI | 2022 | [Pytorch](https://github.com/yuetan031/fedproto) |
 | [Jang J, Ha H, Jung D, Yoon S. Fedclassavg: Local representation learning for personalized federated learning on heterogeneous neural networks. In: Proceedings of the 51st International Conference on Parallel Processing. 2022, 1–10](https://arxiv.org/abs/2210.14226) | Parameter Decoupling | ICPP | 2022 | [Pytorch](https://github.com/hukla/fedclassavg) |
 | [Wang K, He Q, Chen F, Chen C, Huang F, Jin H, Yang Y. Flexifed: Personalized federated learning for edge clients with heterogeneous model architectures. In: Proceedings of the ACM Web Conference 2023. 2023, 2979–2990](https://dl.acm.org/doi/10.1145/3543507.3583347) | Parameter Decoupling | ACM WWW | 2023 | N/A |
 | [Liu C, Yang Y, Cai X, Ding Y, Lu H. Completely heterogeneous federated learning. arXiv preprint arXiv:2210.15865, 2022](https://arxiv.org/abs/2210.15865) | Parameter Decoupling, Knowledge Distillation | arXiv | 2022 | N/A |
 | [Diao E, Ding J, Tarokh V. Heterofl: Computation and communication efficient federated learning for heterogeneous clients. arXiv preprint arXiv:2010.01264, 2020](https://arxiv.org/abs/2010.01264) | Parameter Pruning | ICLR | 2021 | [Pytorch](https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients) |
 | [Qayyum A, Ahmad K, Ahsan M A, Al-Fuqaha A, Qadir J. Collaborative federated learning for health-care: Multi-modal covid-19 diagnosis at the edge. IEEE Open Journal of the Computer Society, 2022, 3: 172–184](https://ieeexplore.ieee.org/document/9891834) | Model Clustering | IEEE | 2022 | N/A |
 | [Li D, Wang J. Fedmd: Heterogenous federated learning via model distillation. arXiv preprint arXiv:1910.03581, 2019](https://arxiv.org/abs/1910.03581) | Knowledge Distillation | NeurIPS | 2019 | N/A |
 | [Huang W, Ye M, Du B, Gao X. Few-shot model agnostic federated learning. In: Proceedings of the 30th ACM International Conference on Multimedia. 2022, 7309–7316](https://dl.acm.org/doi/abs/10.1145/3503161.3548764) | Knowledge Distillation | ACM MM | 2022 | N/A |
 | [Zhang J, Guo S, Guo J, Zeng D, Zhou J, Zomaya A. Towards data-independent knowledge transfer in model-heterogeneous federated learning. IEEE Transactions on Computers, 2023](https://ieeexplore.ieee.org/document/10115052) | Knowledge Distillation | IEEE | 2023 | N/A |
 | [Yang Y, Yang R, Peng H, Li Y, Li T, Liao Y, Zhou P. Fedack: Federated adversarial contrastive knowledge distillation for cross-lingual and cross-model social bot detection. In: Proceedings of the ACM Web Conference 2023. 2023, 1314–1323](https://arxiv.org/abs/2303.07113) | Knowledge Distillation | ACM WWW | 2023 | [Pytorch](https://github.com/846468230/FedACK) |
 | [Niu Z, Wang H, Sun H, Ouyang S, Chen Y w, Lin L. Mckd: Mutually collaborative knowledge distillation for federated domain adaptation and generalization. In: ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2023, 1–5](https://ieeexplore.ieee.org/document/10095699) | Knowledge Distillation | IEEE | 2023 | N/A |
 | [Li Y, Zhou W, Wang H, Mi H, Hospedales T M. Fedh2l: Federated learning with model and statistical heterogeneity. arXiv preprint arXiv:2101.11296, 2021](https://arxiv.org/abs/2101.11296) | Knowledge Distillation | IEEE | 2023 | N/A |
 | [Zhang L, Wu D, Yuan X. Fedzkt: Zero-shot knowledge transfer towards resource-constrained federated learning with heterogeneous on-device models. In: 2022 IEEE 42nd International Conference on Distributed Computing Systems (ICDCS). 2022, 928–938](https://arxiv.org/abs/2109.03775) | Knowledge Distillation | IEEE | 2022 | N/A |

  
### SSFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |
 | [Lin H, Lou J, Xiong L, Shahabi C. Semifed: Semi-supervised federated learning with consistency and pseudo-labeling. arXiv preprint arXiv:2108.09412, 2021](https://arxiv.org/abs/2108.09412) | Instance Augmentation | SIAM | 2021 | N/A |
 | [Lubana E S, Tang C I, Kawsar F, Dick R P, Mathur A. Orchestra: Unsupervised federated learning via globally consistent clustering. arXiv preprint arXiv:2205.11506, 2022](https://arxiv.org/abs/2205.11506) | Instance Augmentation,  Feature Clustering | ICML | 2022 | [Pytorch](https://github.com/akhilmathurs/orchestra) |
 | [Li M, Li Q, Wang Y. Class balanced adaptive pseudo labeling for federated semi-supervised learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023, 16292–16301](https://ieeexplore.ieee.org/document/10203348) | Instance Selection | IEEE | 2023 | N/A |
 | [Wu Z, Li Q, He B. Practical vertical federated learning with unsupervised representation learning. IEEE Transactions on Big Data, 2022](https://arxiv.org/abs/2208.10278) | Feature Augmentation | IEEE | 2022 | [Pytorch](https://github.com/jerrylife/fedonce) |
 | [Kang Y, He Y, Luo J, Fan T, Liu Y, Yang Q. Privacy-preserving federated adversarial domain adaptation over feature groups for interpretability. IEEE Transactions on Big Data, 2022](https://arxiv.org/abs/2111.10934) | Feature Clustering | IEEE | 2022 | N/A |
 |[Castiglia T, Zhou Y, Wang S, Kadhe S, Baracaldo N, Patterson S. Less-vfl: Communication-efficient feature selection for vertical federated learning. arXiv preprint arXiv:2305.02219, 2023](https://arxiv.org/abs/2305.02219) | Feature Selection | ICML | 2023 | N/A |
 | [Feng S. Vertical federated learning-based feature selection with non-overlapping sample utilization. Expert Systems with Applications, 2022, 208: 118097](https://dl.acm.org/doi/10.1016/j.eswa.2022.118097) | Feature Selection | EXPERT SYST APPL | 2022 | N/A |
 | [Jiang J, Burkhalter L, Fu F, Ding B, Du B, Hithnawi A, Li B, Zhang C. Vf-ps: How to select important participants in vertical federated learning, efficiently and securely? Advances in Neural Information Processing Systems, 2022, 35: 2088–2101](https://dl.acm.org/doi/10.5555/3600270.3600422) | Feature Selection, Model Selection; | NIPS | 2022 | 
 | [Liu Q, Yang H, Dou Q, Heng P A. Federated semi-supervised medical image classification via inter-client relation matching. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part III 24. 2021, 325–335](https://arxiv.org/abs/2106.08600) | Consistency Regularization | MICCAI | 2021 | [Pytorch](https://github.com/liuquande/FedIRM) |
 | [Jeong W, Yoon J, Yang E, Hwang S J. Federated semi-supervised learning with inter-client consistency & disjoint learning. arXiv preprint arXiv:2006.12097, 2020](https://arxiv.org/abs/2006.12097) | Consistency Regularization, Parameter Decoupling | ICLR | 2021 | [Tensorflow](https://github.com/wyjeong/FedMatch) |
 | [Shen T, Zhang J, Jia X, Zhang F, Huang G, Zhou P, Kuang K, Wu F, Wu C. Federated mutual learning. arXiv preprint arXiv:2006.16765, 2020](https://arxiv.org/abs/2006.16765) | Domain-dependent Consistency Regularization, Knowledge Distillation | arXiv | 2020 | N/A |
 | [Lin X, Chen H, Xu Y, Xu C, Gui X, Deng Y, Wang Y. Federated learning with positive and unlabeled data. In: International Conference on Machine Learning. 2022, 13344–13355](https://arxiv.org/abs/2106.10904) | Model Weighting | ICML | 2022 | [Python/Torch](https://github.com/littlesunlxy/fedpu-torch) |
 | [Yang D, Xu Z, Li W, Myronenko A, Roth H R, Harmon S, Xu S, Turkbey B, Turkbey E, Wang X, others . Federated semi-supervised learning for covid region segmentation in chest ct using multi-national data from china, italy, japan. Medical image analysis, 2021, 70: 101992](https://arxiv.org/abs/2011.11750) | Model Weighting, Parameter Decoupling | MIA | 2021 | N/A |
 | [Itahara S, Nishio T, Koda Y, Morikura M, Yamamoto K. Distillation-based semi-supervised federated learning for communication-efficient collaborative training with non-iid private data. IEEE Transactions on Mobile Computing, 2021, 22(1): 191–205](https://arxiv.org/abs/2008.06180) | Knowledge Distillation | IEEE | 2021 | N/A |

### USFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

### HOFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

### HEFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

 
