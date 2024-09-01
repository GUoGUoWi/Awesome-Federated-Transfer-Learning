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
#### System
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
#### Incremental
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
 | [Wang K, He Q, Chen F, Chen C, Huang F, Jin H, Yang Y. Flexifed: Personalized federated learning for edge clients with heterogeneous model architectures. In: Proceedings of the ACM Web Conference 2023. 2023, 2979–2990](https://dl.acm.org/doi/10.1145/3543507.3583347) | Parameter Decoupling | ACMWWW | 2023 | N/A |
 | [Liu C, Yang Y, Cai X, Ding Y, Lu H. Completely heterogeneous federated learning. arXiv preprint arXiv:2210.15865, 2022](https://arxiv.org/abs/2210.15865) | Parameter Decoupling, Knowledge Distillation | arXiv | 2022 | N/A |
 | [Diao E, Ding J, Tarokh V. Heterofl: Computation and communication efficient federated learning for heterogeneous clients. arXiv preprint arXiv:2010.01264, 2020](https://arxiv.org/abs/2010.01264) | Parameter Pruning | ICLR | 2021 | [Pytorch](https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients) |
 | [Qayyum A, Ahmad K, Ahsan M A, Al-Fuqaha A, Qadir J. Collaborative federated learning for health-care: Multi-modal covid-19 diagnosis at the edge. IEEE Open Journal of the Computer Society, 2022, 3: 172–184](https://ieeexplore.ieee.org/document/9891834) | Model Clustering | IEEE | 2022 | N/A |
 | [Li D, Wang J. Fedmd: Heterogenous federated learning via model distillation. arXiv preprint arXiv:1910.03581, 2019](https://arxiv.org/abs/1910.03581) | Knowledge Distillation | NeurIPS | 2019 | N/A |
 | [Huang W, Ye M, Du B, Gao X. Few-shot model agnostic federated learning. In: Proceedings of the 30th ACM International Conference on Multimedia. 2022, 7309–7316](https://dl.acm.org/doi/abs/10.1145/3503161.3548764) | Knowledge Distillation | ACMMM | 2022 | N/A |
 | [Zhang J, Guo S, Guo J, Zeng D, Zhou J, Zomaya A. Towards data-independent knowledge transfer in model-heterogeneous federated learning. IEEE Transactions on Computers, 2023](https://ieeexplore.ieee.org/document/10115052) | Knowledge Distillation | IEEE | 2023 | N/A |
 | [Yang Y, Yang R, Peng H, Li Y, Li T, Liao Y, Zhou P. Fedack: Federated adversarial contrastive knowledge distillation for cross-lingual and cross-model social bot detection. In: Proceedings of the ACM Web Conference 2023. 2023, 1314–1323](https://arxiv.org/abs/2303.07113) | Knowledge Distillation | ACMWWW | 2023 | [Pytorch](https://github.com/846468230/FedACK) |
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
 | [Jiang J, Burkhalter L, Fu F, Ding B, Du B, Hithnawi A, Li B, Zhang C. Vf-ps: How to select important participants in vertical federated learning, efficiently and securely? Advances in Neural Information Processing Systems, 2022, 35: 2088–2101](https://dl.acm.org/doi/10.5555/3600270.3600422) | Feature Selection, Model Selection; | NIPS | 2022 | N/A |
 | [Liu Q, Yang H, Dou Q, Heng P A. Federated semi-supervised medical image classification via inter-client relation matching. In: Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part III 24. 2021, 325–335](https://arxiv.org/abs/2106.08600) | Consistency Regularization | MICCAI | 2021 | [Pytorch](https://github.com/liuquande/FedIRM) |
 | [Jeong W, Yoon J, Yang E, Hwang S J. Federated semi-supervised learning with inter-client consistency & disjoint learning. arXiv preprint arXiv:2006.12097, 2020](https://arxiv.org/abs/2006.12097) | Consistency Regularization, Parameter Decoupling | ICLR | 2021 | [Tensorflow](https://github.com/wyjeong/FedMatch) |
 | [Shen T, Zhang J, Jia X, Zhang F, Huang G, Zhou P, Kuang K, Wu F, Wu C. Federated mutual learning. arXiv preprint arXiv:2006.16765, 2020](https://arxiv.org/abs/2006.16765) | Domain-dependent Consistency Regularization, Knowledge Distillation | arXiv | 2020 | N/A |
 | [Lin X, Chen H, Xu Y, Xu C, Gui X, Deng Y, Wang Y. Federated learning with positive and unlabeled data. In: International Conference on Machine Learning. 2022, 13344–13355](https://arxiv.org/abs/2106.10904) | Model Weighting | ICML | 2022 | [Python/Torch](https://github.com/littlesunlxy/fedpu-torch) |
 | [Yang D, Xu Z, Li W, Myronenko A, Roth H R, Harmon S, Xu S, Turkbey B, Turkbey E, Wang X, others . Federated semi-supervised learning for covid region segmentation in chest ct using multi-national data from china, italy, japan. Medical image analysis, 2021, 70: 101992](https://arxiv.org/abs/2011.11750) | Model Weighting, Parameter Decoupling | MIA | 2021 | N/A |
 | [Itahara S, Nishio T, Koda Y, Morikura M, Yamamoto K. Distillation-based semi-supervised federated learning for communication-efficient collaborative training with non-iid private data. IEEE Transactions on Mobile Computing, 2021, 22(1): 191–205](https://arxiv.org/abs/2008.06180) | Knowledge Distillation | IEEE | 2021 | N/A |

### USFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |
 | [Liu Y, Guo S, Zhang J, Zhou Q, Wang Y, Zhao X. Feature correlation-guided knowledge transfer for federated self-supervised learning. arXiv preprint arXiv:2211.07364, 2022](https://arxiv.org/abs/2211.07364) | Feature Alignment | arXiv | 2022 | N/A |
 | [Zhang F, Kuang K, You Z, Shen T, Xiao J, Zhang Y, Wu C, Zhuang Y, Li X. Federated unsupervised representation learning. arXiv e-prints, 2020, arXiv–2010](https://arxiv.org/abs/2010.08982) | Feature Selection | FITEE | 2023 | N/A |
 | [Zhang X, Mavromatics A, Vafeas A, Nejabati R, Simeonidou D. Federated feature selection for horizontal federated learning in iot networks. IEEE Internet of Things Journal, 2023](https://ieeexplore.ieee.org/document/10017376) |  Feature Alignment | IEEE | 2023 | N/A |
 | [Zhuang W, Gan X, Wen Y, Zhang S, Yi S. Collaborative unsupervised visual representation learning from decentralized data. In: Proceedings of the IEEE/CVF international conference on computer vision. 2021, 4912–4921](https://arxiv.org/abs/2108.06492) | Parameter Decoupling | IEEE | 2021 | [Python](https://github.com/EasyFL-AI/EasyFL/tree/master/applications/fedssl) |
 | [Zhuang W, Wen Y, Zhang S. Joint optimization in edge-cloud continuum for federated unsupervised person re-identification. In: Proceedings of the 29th ACM International Conference on Multimedia. 2021, 433–441](https://arxiv.org/abs/2108.06493) | Model Weighting | ACMMM | 2021 | N/A |
 | [Liang X, Lin Y, Fu H, Zhu L, Li X. Rscfed: Random sampling consensus federated semi-supervised learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022, 10154–10163](https://arxiv.org/abs/2203.13993) | Model Weighting, Model Clustering | CVPR | 2022 | [Pytorch](https://github.com/xmed-lab/rscfed) |
 | [Beilharz J, Pfitzner B, Schmid R, Geppert P, Arnrich B, Polze A. Implicit model specialization through dag-based decentralized federated learning. In: Proceedings of the 22nd International Middleware Conference. 2021, 310–322](https://arxiv.org/abs/2111.01257) | Model Selection | IFIP | 2021 | [TensorFlow](https://github.com/osmhpi/federated-learning-dag) |
 | [Li S, Zhou T, Tian X, Tao D. Learning to collaborate in decentralized learning of personalized models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022, 9766–9775](https://ieeexplore.ieee.org/document/9880456) | Model Selection | IEEE | 2022 | N/A |
 | [Zhuang W, Wen Y, Zhang S. Divergence-aware federated self-supervised learning. arXiv preprint arXiv:2204.04385, 2022](https://arxiv.org/abs/2204.04385) | Model Interpolation | ICLR | 2022 | [Python](https://github.com/EasyFL-AI/EasyFL/tree/master/applications/fedssl) |
 | [Han S, Park S, Wu F, Kim S, Wu C, Xie X, Cha M. Fedx: Unsupervised federated learning with cross knowledge distillation. In: European Conference on Computer Vision. 2022, 691–707](https://arxiv.org/abs/2207.09158) | Knowledge Distillation | ECCV | 2022 | [Pytorch](https://github.com/sungwon-han/fedx) |

### HOFTL
#### Prior Shift
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |
 

#### Covariate Shift
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |
 | [Chen H, Frikha A, Krompass D, Gu J, Tresp V. Fraug: Tackling federated learning with non-iid features via representation augmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023, 4849–4859](https://arxiv.org/abs/2205.14900) | Feature Augmentation | ICCV | 2023 | [Python](https://github.com/HaokunChen245/FRAug) |
 | [Zhou T, Konukoglu E. Fedfa: Federated feature augmentation. arXiv preprint arXiv:2301.12995, 2023](https://arxiv.org/abs/2301.12995) | Feature Augmentation | ICLR | 2023 | [Pytorch](https://github.com/tfzhou/fedfa) |
 | [Huang W, Ye M, Shi Z, Li H, Du B. Rethinking federated learning with domain shift: A prototype view. In: 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2023, 16312–16322](https://ieeexplore.ieee.org/document/10203389) | Feature Clustering, Consistency Regularization | CVPR | 2023 | N/A |
 | [Liu B, Guo Y, Chen X. Pfa: Privacy-preserving federated adaptation for effective model personalization. In: Proceedings of the Web Conference 2021. 2021, 923–934](https://arxiv.org/abs/2103.01548) | Feature Clustering, Model Clustering | ACMWWW | 2021 | [Pytorch](https://github.com/lebyni/PFA) |
 | [Liu Q, Chen C, Qin J, Dou Q, Heng P A. Feddg: Federated domain generalization on medical image segmentation via episodic learning in continuous frequency space. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021, 1013–1023](https://arxiv.org/abs/2103.06030) | Feature Alignment | CVPR | 2021 | [Pytorch](https://github.com/liuquande/FedDG-ELCFS) |
 | [Li X, Jiang M, Zhang X, Kamp M, Dou Q. Fedbn: Federated learning on non-iid features via local batch normalization. arXiv preprint arXiv:2102.07623, 2021](https://arxiv.org/abs/2102.07623) | Consistency Regularization | ICLR | 2021 | [Pytorch](https://github.com/med-air/FedBN) |
 | [Wang H, Zhao H, Wang Y, Yu T, Gu J, Gao J. Fedkc: Federated knowledge composition for multi-lingual natural language understanding. In: Proceedings of the ACM Web Conference 2022. 2022, 1839–1850](https://dl.acm.org/doi/abs/10.1145/3485447.3511988) | Domain-dependent Consistency Regularization | ACMWWW | 2022 | N/A |
 | [Wang K, Mathews R, Kiddon C, Eichner H, Beaufays F, Ramage D. Federated evaluation of on-device personalization. arXiv preprint arXiv:1910.10252, 2019](https://arxiv.org/abs/1910.10252) | Prior Shift | arXiv | 2019 | N/A |
 |  [Li T, Sahu A K, Zaheer M, Sanjabi M, Talwalkar A, Smith V. Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2020, 2: 429–450](https://arxiv.org/abs/1812.06127) | Parameter Restriction | MLSys | 2020 | [Pytorch](https://github.com/litian96/FedProx) |
 | [Dinh C T, Tran N H, Nguyen T D, Bao W, Zomaya A Y, Zhou B B. Federated learning with proximal stochastic variance reduced gradient algorithms. In: Proceedings of the 49th International Conference on Parallel Processing. 2020, 1–11](https://dl.acm.org/doi/abs/10.1145/3404397.3404457) | Parameter Restriction | ICPP | 2020 | N/A |
 | [Li T, Hu S, Beirami A, Smith V. Ditto: Fair and robust federated learning through personalization. In: International Conference on Machine Learning. 2021, 6357–6368](https://arxiv.org/abs/2012.04221) | Parameter Restriction | ICML | 2021 | [Pytorch](https://github.com/litian96/ditto) |
 | [Deng Y, Kamani M M, Mahdavi M. Adaptive personalized federated learning. arXiv preprint arXiv:2003.13461, 2020](https://arxiv.org/abs/2003.13461) | Parameter Restriction, Model Interpolation | arXiv | 2020 | [Pytorch](https://github.com/MLOPTPSU/FedTorch) |
 | [Pillutla K, Malik K, Mohamed A R, Rabbat M, Sanjabi M, Xiao L. Federated learning with partial model personalization. In: International Conference on Machine Learning. 2022, 17716–17758](https://arxiv.org/abs/2204.03809) | Parameter Decoupling | ICML | 2022 | [Pytorch](https://github.com/krishnap25/fl_partial_personalization) |
 | [Li A, Sun J, Li P, Pu Y, Li H, Chen Y. Hermes: an efficient federated learning framework for heterogeneous mobile clients. In: Proceedings of the 27th Annual International Conference on Mobile Computing and Networking. 2021, 420–437](https://dl.acm.org/doi/10.1145/3447993.3483278) | Parameter Decoupling | ACM MobiCom | 2021 | N/A |
 | [Liu C, Yang Y, Cai X, Ding Y, Lu H. Completely heterogeneous federated learning. arXiv preprint arXiv:2210.15865, 2022](https://arxiv.org/abs/2210.15865) | Parameter Decoupling, Knowledge Distillation | arXiv | 2022 | N/A |
 | [Chen H Y, Chao W L. On bridging generic and personalized federated learning for image classification. arXiv preprint arXiv:2107.00778, 2021](https://arxiv.org/abs/2107.00778) | Parameter Decoupling, Parameter Restriction, Model Interpolation | ICLR | 2022 | [Pytorch](https://github.com/hongyouc/fed-rod) |
 | [Zhuang W, Wen Y, Zhang S. Joint optimization in edge-cloud continuum for federated unsupervised person re-identification. In: Proceedings of the 29th ACM International Conference on Multimedia. 2021, 433–441](https://arxiv.org/abs/2108.06493) | Model Weighting | ACMMM | 2021 | N/A |
 | [Zhang R, Xu Q, Yao J, Zhang Y, Tian Q, Wang Y. Federated domain generalization with generalization adjustment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023, 3954–3963](https://ieeexplore.ieee.org/document/10203192) | Model Weighting | CVPR | 2023 | [Pytorch](https://github.com/MediaBrain-SJTU/FedDG-GA) |
 | [Liu X, Xi W, Li W, Xu D, Bai G, Zhao J. Co-mda: Federated multi-source domain adaptation on black-box models. IEEE Transactions on Circuits and Systems for Video Technology, 2023](https://ieeexplore.ieee.org/document/10128163) | Model Weighting | IEEE | 2023 | N/A |
 | [Zhu J, Ma X, Blaschko M B. Confidence-aware personalized federated learning via variational expectation maximization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023, 24542–24551](https://arxiv.org/abs/2305.12557) | Model Weighting, Consistency Regularization | CVPR | 2023 | [Pytorch](https://github.com/junyizhu-ai/confidence_aware_pfl) |
 | [Yang D, Xu Z, Li W, Myronenko A, Roth H R, Harmon S, Xu S, Turkbey B, Turkbey E, Wang X, others . Federated semi-supervised learning for covid region segmentation in chest ct using multi-national data from china, italy, japan. Medical image analysis, 2021, 70: 101992](https://arxiv.org/abs/2011.11750) | Model Weighting, Parameter Decoupling | MIA | 2021 | N/A |
 | [Beilharz J, Pfitzner B, Schmid R, Geppert P, Arnrich B, Polze A. Implicit model specialization through dag-based decentralized federated learning. In: Proceedings of the 22nd International Middleware Conference. 2021, 310–322](https://arxiv.org/abs/2111.01257) | Model Selection | IFIP | 2021 | [TensorFlow](https://github.com/osmhpi/federated-learning-dag) |
 | [Luping W, Wei W, Bo L. Cmfl: Mitigating communication overhead for federated learning. In: 2019 IEEE 39th international conference on distributed computing systems (ICDCS). 2019, 954–964](https://ieeexplore.ieee.org/document/8885054) | Model Selection | ICDCS | 2019 | N/A |
 | [Qayyum A, Ahmad K, Ahsan M A, Al-Fuqaha A, Qadir J. Collaborative federated learning for health-care: Multi-modal covid-19 diagnosis at the edge. IEEE Open Journal of the Computer Society, 2022, 3: 172–184](https://ieeexplore.ieee.org/document/9891834) | Model Clustering | IEEE | 2022 | N/A |
 | [Xie H, Xiong L, Yang C. Federated node classification over graphs with latent link-type heterogeneity. In: Proceedings of the ACM Web Conference 2023. 2023, 556–566](https://dl.acm.org/doi/abs/10.1145/3543507.3583471) | Model Clustering | ACMWWW | 2023 | [Python](https://github.com/Oxfordblue7/FedLIT) |
 | [Wang H, Yurochkin M, Sun Y, Papailiopoulos D, Khazaeni Y. Federated learning with matched averaging. arXiv preprint arXiv:2002.06440, 2020](http://www.arxiv.org/abs/2002.06440) | Model Clustering | ICLR | 2020 | [Pytorch](https://github.com/IBM/FedMA) |
 | [Ruan Y, Joe-Wong C. Fedsoft: Soft clustered federated learning with proximal local updating. In: Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 8124–8131](https://arxiv.org/abs/2112.06053) | Model Clustering | AAAI | 2022 | N/A |
 | [Marfoq O, Neglia G, Vidal R, Kameni L. Personalized federated learning through local memorization. In: International Conference on Machine Learning. 2022, 15070–15092](https://arxiv.org/abs/2111.09360) | Model Interpolation | ICML | 2022 | [Pytorch](https://github.com/omarfoq/knn-per) |
 | [Li D, Wang J. Fedmd: Heterogenous federated learning via model distillation. arXiv preprint arXiv:1910.03581, 2019](https://arxiv.org/abs/1910.03581) | Knowledge Distillation | NeurIPS | 2019 | N/A |
 | [Li Y, Zhou W, Wang H, Mi H, Hospedales T M. Fedh2l: Federated learning with model and statistical heterogeneity. arXiv preprint arXiv:2101.11296, 2021](https://arxiv.org/abs/2101.11296) | Knowledge Distillation | IEEE | 2023 | N/A |
 | [Wu Y, Kang Y, Luo J, He Y, Yang Q. Fedcg: Leverage conditional gan for protecting privacy and maintaining competitive performance in federated learning. arXiv preprint arXiv:2111.08211, 2021](https://arxiv.org/abs/2111.08211) | Knowledge Distillation | IJCAI | 2022 | [Pytorch](https://github.com/FederatedAI/research/tree/main/publications/FedCG) |

#### Feature Concept Shift
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |
 | [Ghosh A, Chung J, Yin D, Ramchandran K. An efficient framework for clustered federated learning. Advances in Neural Information Processing Systems, 2020, 33: 19586–19597](https://arxiv.org/abs/2006.04088) | Model Clustering | NeurIPS | 2020 | [Pytorch](https://github.com/jichan3751/ifca) |

#### Label Concept Shift
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

#### Quantity Shift
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |


### HEFTL
 | Paper | Strategy | Venue | Year | Code |
 | :--- | :---: | :---: | :---: | :---: |

 
