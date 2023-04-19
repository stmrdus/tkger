# TKGER
Some papers on Temporal Knowledge Graph Embedding and Reasoning

## Datasets

| Name  | #Entities  | #Relations  | #Timestamps  | #Collections  |  Timestamp | Link download  |
|---|---|---|---|---|---|---|
| ICEWS14   | 7128  | 230  | 365  | 90730  | point  | https://paperswithcode.com/sota/link-prediction-on-icews14-1  |
| ICEWS05-15  | 10488  | 251 | 4017  | 479329  | point  |  https://paperswithcode.com/sota/link-prediction-on-icews05-15-1 |
| ICEWS18   | 23033  | 256  | 304  |  468558 |  point |  https://docs.dgl.ai/en/0.8.x/generated/dgl.data.ICEWS18Dataset.html |
| GDELT  |  500 |  20 | 366  | 3419607  | point  | https://www.gdeltproject.org/  |
| YAGO15k  | 15403  | 32  | 169  | 138048  |  interval | https://paperswithcode.com/sota/link-prediction-on-yago15k-1  |
| WIKIDATA  | 11153  | 96  | 328  | 150079  |  interval | https://www.wikidata.org/wiki/Wikidata:Main_Page  |

## [Content](#content)

<table>
<tr><td colspan="2"><a href="#survey-papers">1. Survey</a></td></tr>
<tr><td colspan="2"><a href="#approaches">2. Approaches</a></td></tr> 


### 2023

[26] Zhang, S., Liang, X., Li, Z., Feng, J., Zheng, X., & Wu, B. (2023, April). BiQCap: A Biquaternion and Capsule Network-Based Embedding Model for Temporal Knowledge Graph Completion. In Database Systems for Advanced Applications: 28th International Conference, DASFAA 2023, Tianjin, China, April 17–20, 2023, Proceedings, Part II (pp. 673-688). Cham: Springer Nature Switzerland.

[25] Mo, C., Wang, Y., Jia, Y., & Luo, C. (2023, April). Time-aware Quaternion Convolutional Network for Temporal Knowledge Graph Reasoning. In Neural Information Processing: 29th International Conference, ICONIP 2022, Virtual Event, November 22–26, 2022, Proceedings, Part IV (pp. 300-312). Singapore: Springer Nature Singapore.

[24] Zheng, S., Yin, H., Chen, T., Nguyen, Q. V. H., Chen, W., & Zhao, L. (2023). DREAM: Adaptive Reinforcement Learning based on Attention Mechanism for Temporal Knowledge Graph Reasoning. arXiv preprint arXiv:2304.03984.

[23] Yue, L., Ren, Y., Zeng, Y., Zhang, J., Zeng, K., & Wan, J. (2023, April). Block Decomposition with Multi-granularity Embedding for Temporal Knowledge Graph Completion. In Database Systems for Advanced Applications: 28th International Conference, DASFAA 2023, Tianjin, China, April 17–20, 2023, Proceedings, Part II (pp. 706-715). Cham: Springer Nature Switzerland.

[22] Gong, X., Qin, J., Chai, H., Ding, Y., Jia, Y., & Liao, Q. (2023, April). Temporal-Relational Matching Network for Few-Shot Temporal Knowledge Graph Completion. In Database Systems for Advanced Applications: 28th International Conference, DASFAA 2023, Tianjin, China, April 17–20, 2023, Proceedings, Part II (pp. 768-783). Cham: Springer Nature Switzerland.

[21] Liu, W., Wang, P., Zhang, Z., & Liu, Q. (2023). Multi-Scale Convolutional Neural Network for Temporal Knowledge Graph Completion. Cognitive Computation, 1-7.

[20] Zhang, S., Liang, X., Tang, H., Zheng, X., Zhang, A. X., & Ma, Y. DuCape: Dual Quaternion and Capsule Network Based Temporal Knowledge Graph Embedding. ACM Transactions on Knowledge Discovery from Data.

[19] Nie, H., Zhao, X., Yao, X., Jiang, Q., Bi, X., Ma, Y., & Sun, Y. (2023). Temporal-structural importance weighted graph convolutional network for temporal knowledge graph completion. Future Generation Computer Systems.

[18] Wang, X., Lyu, S., Wang, X., Wu, X., & Chen, H. (2023). Temporal knowledge graph embedding via sparse transfer matrix. Information Sciences, 623, 56-69.

[17] Bai, L., Yu, W., Chai, D., Zhao, W., & Chen, M. (2023). Temporal knowledge graphs reasoning with iterative guidance by temporal logical rules. Information Sciences, 621, 22-35.

[16] Nolting, S., Han, Z., & Tresp, V. (2023). Modeling the evolution of temporal knowledge graphs with uncertainty. arXiv preprint arXiv:2301.04977.

[15] Gottschalk, S., Kacupaj, E., Abdollahi, S., Alves, D., Amaral, G., Koutsiana, E., ... & Thakkar, G. (2023). Oekg: The open event knowledge graph. arXiv preprint arXiv:2302.14688.

[14] Hou, X., Ma, R., Yan, L., & Ma, Z. (2023). DAuCNet: deep autoregressive framework for temporal link prediction combining copy mechanism network. Knowledge and Information Systems, 1-25.

[13] Shao, P., Liu, T., Che, F., Zhang, D., & Tao, J. (2023). Adaptive pseudo-Siamese policy network for temporal knowledge prediction. Neural Networks.

[12] Gao, Q., Wang, W., Huang, L., Yang, X., Li, T., & Fujita, H. (2023). Dual-grained human mobility learning for location-aware trip recommendation with spatial–temporal graph knowledge fusion. Information Fusion, 92, 46-63.

[11] Zhong, Y., & Huang, C. (2023). A dynamic graph representation learning based on temporal graph transformer. Alexandria Engineering Journal, 63, 359-369.

[10] Zhao, N., Long, Z., Wang, J., & Zhao, Z. D. (2023). AGRE: A knowledge graph recommendation algorithm based on multiple paths embeddings RNN encoder. Knowledge-Based Systems, 259, 110078.

[9] Li, Z., Yu, J., Zhang, G., & Xu, L. (2023). Dynamic spatio-temporal graph network with adaptive propagation mechanism for multivariate time series forecasting. Expert Systems with Applications, 216, 119374.

[8] Wang, H., Cai, S., Liu, P., Zhang, J., Shen, Z., & Liu, K. (2023). DP-STGAT: Traffic statistics publishing with differential privacy and a spatial-temporal graph attention network. Information Sciences, 623, 258-274.

[7] Wen, H., Lin, Y., Xia, Y., Wan, H., Zimmermann, R., & Liang, Y. (2023). DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with Denoising Diffusion Models. arXiv preprint arXiv:2301.13629.

[6] Mo, X., Tang, R., & Liu, H. (2023). A relation-aware heterogeneous graph convolutional network for relationship prediction. Information Sciences, 623, 311-323.

[5] Lou, Y., Wang, C., Gu, T., Feng, H., Chen, J., & Yu, J. X. (2023). Time-topology analysis on temporal graphs. The VLDB Journal, 1-29.

[4] Wang, J., Shi, Y., Yu, H., Zhang, K., Wang, X., Yan, Z., & Li, H. (2023). Temporal Density-aware Sequential Recommendation Networks with Contrastive Learning. Expert Systems with Applications, 211, 118563.

[3] Huan, C., Song, S. L., Pandey, S., Liu, H., Liu, Y., Lepers, B., ... & Wu, Y. (2023). TEA: A General-Purpose Temporal Graph Random Walk Engine.

[2] Huang, N., Wang, S., Wang, R., Cai, G., Liu, Y., & Dai, Q. (2023). Gated spatial-temporal graph neural network based short-term load forecasting for wide-area multiple buses. International Journal of Electrical Power & Energy Systems, 145, 108651.

[1] Li, Y., Chen, H., Li, Y., Li, L., Philip, S. Y., & Xu, G. (2023). Reinforcement Learning based Path Exploration for Sequential Explainable Recommendation. IEEE Transactions on Knowledge and Data Engineering.

### 2022

[1] (BTDG) Yujing Lai, Chuan Chen, Zibin Zheng, Yangqing Zhang. ["Block term decomposition with distinct time granularities for temporal knowledge graph completion"](https://www.sciencedirect.com/science/article/abs/pii/S0957417422004511?via%3Dihub). Expert Systems with Applications 2022.

[2] (EvoExplore) Jiasheng Zhang, Shuang Liang, Yongpan Sheng, Jie Shao. ["Temporal knowledge graph representation learning with local and global evolutions"](https://www.sciencedirect.com/science/article/abs/pii/S0950705122006141?via%3Dihub). Knowledge-Based Systems 2022.

[3] (TuckERT) Pengpeng Shao, Dawei Zhang, Guohua Yang, Jianhua Tao, Feihu Che, Tong Liu. ["Tucker decomposition-based temporal knowledge graph completion"](https://www.sciencedirect.com/science/article/abs/pii/S0950705121010303?via%3Dihub). Knowledge Based Systems 2022. 

[4] (BoxTE) Johannes Messner, Ralph Abboud, Ismail Ilkan Ceylan. ["Temporal Knowledge Graph Completion Using Box Embeddings"](https://ojs.aaai.org/index.php/AAAI/article/view/20746). AAAI 2022.

[5] (TempoQR) Costas Mavromatis, Prasanna Lakkur Subramanyam, Vassilis N. Ioannidis, Adesoji Adeshina, Phillip R. Howard, Tetiana Grinberg, Nagib Hakim, George Karypis. ["TempoQR: Temporal Question Reasoning over Knowledge Graphs"](https://ojs.aaai.org/index.php/AAAI/article/view/20526). AAAI 2022. https://github.com/cmavro/TempoQR

[6] (TLogic) Yushan Liu, Yunpu Ma, Marcel Hildebrandt, Mitchell Joblin, Volker Tresp. ["TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs"](https://ojs.aaai.org/index.php/AAAI/article/view/20330). AAAI 2022. https://github.com/liu-yushan/TLogic

[7] (MetaTKGR) Ruijie Wang, zheng li, Dachun Sun, Shengzhong Liu, Jinning Li, Bing Yin, Tarek Abdelzaher. ["Learning to Sample and Aggregate: Few-shot Reasoning over Temporal Knowledge Graphs"](https://openreview.net/forum?id=1LmgISIDZJ). NeurIPS 2022.

[8] (CEN) Zixuan Li, Saiping Guan, Xiaolong Jin, Weihua Peng, Yajuan Lyu, Yong Zhu, Long Bai, Wei Li, Jiafeng Guo, Xueqi Cheng. ["Complex Evolutional Pattern Learning for Temporal Knowledge Graph Reasoning"](https://aclanthology.org/2022.acl-short.32/). ACL 2022. https://github.com/lee-zix/cen

[9] (RotateQVS) Kai Chen, Ye Wang, Yitong Li, Aiping Li. ["RotateQVS: Representing Temporal Information as Rotations in Quaternion Vector Space for Temporal Knowledge Graph Completion"](https://aclanthology.org/2022.acl-long.402/). ACL 2022. 

[10] (rGalT) Yifu Gao, Linhui Feng, Zhigang Kan, Yi Han, Linbo Qiao, Dongsheng Li. ["Modeling Precursors for Temporal Knowledge Graph Reasoning via Auto-encoder Structure"](https://www.ijcai.org/proceedings/2022/284). IJCAI 2022.

[11] (TiRGN) Yujia Li, Shiliang Sun, Jing Zhao. ["TiRGN: Time-Guided Recurrent Graph Network with Local-Global Historical Patterns for Temporal Knowledge Graph Reasoning"](https://www.ijcai.org/proceedings/2022/299). IJCAI 2022. https://github.com/Liyyy2122/TiRGN

[12] (ALRE-IR) Xin Mei∗, Libin Yang∗, Zuowei Jiang, Xiaoyan Cai. ["An Adaptive Logical Rule Embedding Model for Inductive Reasoning over Temporal Knowledge Graphs"](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.493/). EMNLP 2022. 

[13] (TKGC-AGP) Linhai Zhang, Deyu Zhou. ["Temporal Knowledge Graph Completion with Approximated Gaussian Process Embedding"](https://aclanthology.org/2022.coling-1.416/). COLING 2022.

[14] (DA-Net) Kangzheng Liu, Feng Zhao, Hongxu Chen, Yicong Li, Guandong Xu, Hai Jin. ["DA-Net: Distributed Attention Network for Temporal Knowledge Graph Reasoning"](https://dl.acm.org/doi/10.1145/3511808.3557280). CIKM 2022.

[15] (TLT-KGE) Fuwei Zhang, Zhao Zhang, Xiang Ao, Fuzhen Zhuang, Yongjun Xu, Qing He. ["Along the Time: Timeline-traced Embedding for Temporal Knowledge Graph Completion"](https://dl.acm.org/doi/10.1145/3511808.3557233). CIKM 2022.

[16] EvoKG) Namyong Park, Fuchen Liu, Purvanshi Mehta, Dana Cristofor, Christos Faloutsos, Yuxiao Dong. ["EvoKG: Jointly Modeling Event Time and Network Structure for Reasoning over Temporal Knowledge Graphs"](https://dl.acm.org/doi/10.1145/3488560.3498451). WSDM 2022. https://github.com/NamyongPark/EvoKG

[17] (ARIM-TE) Tingyi Zhang, Zhixu Li, Jiaan Wang, Jianfeng Qu, Lin Yuan, An Liu, Lei Zhao, Zhigang Chen. ["Aligning Internal Regularity and External Influence of Multi-granularity for Temporal Knowledge Graph Embedding"](https://link.springer.com/chapter/10.1007/978-3-031-00129-1_10). DASFAA 2022. 

[18] (TRHyTE) Lin Yuan, Zhixu Li, Jianfeng Qu, Tingyi Zhang, An Liu, Lei Zhao, Zhigang Chen. ["TRHyTE: Temporal Knowledge Graph Embedding Based on Temporal-Relational Hyperplanes"](https://link.springer.com/chapter/10.1007/978-3-031-00123-9_10). DASFAA 2022. 

[19] (SANe) Yancong Li, Xiaoming Zhang, Bo Zhang, Haiying Ren. ["Each Snapshot to Each Space: Space Adaptation for Temporal Knowledge Graph Completion"](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_15). ISWC 2022. 

[20] (ST-KGE) Mojtaba Nayyeri, Sahar Vahdati, Md Tansen Khan, Mirza Mohtashim Alam, Lisa Wenige, Andreas Behrend, Jens Lehmann. ["Dihedron Algebraic Embeddings for Spatio-Temporal Knowledge Graph Completion"](https://link.springer.com/chapter/10.1007/978-3-031-06981-9_15). ESWC 2022. 

### 2021

[1] (TPath) Luyi Bai, Wenting Yu, Mingzhuo Chen, Xiangnan Ma. ["Multi-hop reasoning over paths in temporal knowledge graphs using reinforcement learning"](https://www.sciencedirect.com/science/article/abs/pii/S1568494621000673?via%3Dihub). Applied Soft Computing 2021. 

[2] (xERTE) Zhen Han, Peng Chen, Yunpu Ma, Volker Tresp. ["Explainable Subgraph Reasoning for Forecasting on Temporal Knowledge Graphs"](https://iclr.cc/virtual/2021/poster/3378). ICLR 2021. https://github.com/TemporalKGTeam/xERTE

[3] (ChronoR) Ali Sadeghian, Mohammadreza Armandpour, Anthony Colas, Daisy Zhe Wang. ["ChronoR: Rotation Based Temporal Knowledge Graph Embedding"](https://ojs.aaai.org/index.php/AAAI/article/view/16802). AAAI 2021.

[4] (CyGNet) Cunchao Zhu, Muhao Chen, Changjun Fan, Guangquan Cheng, Yan Zhang. ["Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Networks"](https://ojs.aaai.org/index.php/AAAI/article/view/16604). AAAI 2021. https://github.com/CunchaoZ/CyGNet

[5] (NLSM) Tony Gracious, Shubham Gupta, Arun Kanthali, Rui M. Castro, Ambedkar Dukkipati. ["Neural Latent Space Model for Dynamic Networks and Temporal Knowledge Graphs"](https://ojs.aaai.org/index.php/AAAI/article/view/16526). AAAI 2021. 

[6] (CluSTeR) Zixuan Li, Xiaolong Jin, Saiping Guan, Wei Li, Jiafeng Guo, Yuanzhuo Wang, Xueqi Cheng. ["Search from History and Reason for Future: Two-stage Reasoning on Temporal Knowledge Graphs"](https://aclanthology.org/2021.acl-long.365/). ACL/IJCNLP 2021. 

[7] (HERCULES) Sebastien Montella, Lina Maria Rojas-Barahona, Johannes Heinecke. ["Hyperbolic Temporal Knowledge Graph Embeddings with Relational and Time Curvatures"](https://aclanthology.org/2021.findings-acl.292/). ACL/IJCNLP (Findings) 2021. 

[8] (HIPNet) Yongquan He, Peng Zhang, Luchen Liu, Qi Liang, Wenyuan Zhang, Chuang Zhang, ["HIP Network: Historical Information Passing Network for Extrapolation Reasoning on Temporal Knowledge Graph"](https://www.ijcai.org/proceedings/2021/264). IJCAI 2021. https://github.com/Yongquan-He/HIP-network

[9] (TANGO) Zhen Han, Zifeng Ding, Yunpu Ma, Yujia Gu, Volker Tresp. ["Learning Neural Ordinary Equations for Forecasting Future Links on Temporal Knowledge Graphs"](https://aclanthology.org/2021.emnlp-main.658/). EMNLP 2021.

[10] (TEA-GNN) Chengjin Xu, Fenglong Su, Jens Lehmann. ["Time-aware Graph Neural Network for Entity Alignment between Temporal Knowledge Graphs"](https://aclanthology.org/2021.emnlp-main.709/). EMNLP 2021. https://github.com/soledad921/TEA-GNN

[11] (TEE) Zhen Han, Gengyuan Zhang, Yunpu Ma, Volker Tresp. ["Time-dependent Entity Embedding is not All You Need: A Re-evaluation of Temporal Knowledge Graph Completion Models under a Unified Framework"](https://aclanthology.org/2021.emnlp-main.639/). EMNLP 2021. 

[12] (TITer) Haohai Sun, Jialun Zhong, Yunpu Ma, Zhen Han, Kun He. ["TimeTraveler: Reinforcement Learning for Temporal Knowledge Graph Forecasting"](https://aclanthology.org/2021.emnlp-main.655/). EMNLP 2021. https://github.com/JHL-HUST/TITer

[13] (RTFE) Youri Xu, Haihong E, Meina Song, Wenyu Song, Xiaodong Lv, Haotian Wang, Jinrui Yang. ["RTFE: A Recursive Temporal Fact Embedding Framework for Temporal Knowledge Graph Completion"](https://www.aclweb.org/anthology/2021.naacl-main.451/). NAACL-HLT 2021.

[14] (TeLM) Chengjin Xu, Yung-Yu Chen, Mojtaba Nayyeri, Jens Lehmann. ["Temporal Knowledge Graph Completion using a Linear Temporal Regularizer and Multivector Embeddings"](https://www.aclweb.org/anthology/2021.naacl-main.202/). NAACL-HLT 2021. https://github.com/soledad921/TeLM

[15] (T-GAP) Jaehun Jung, Jinhong Jung, U. Kang. ["Learning to Walk across Time for Interpretable Temporal Knowledge Graph Completion"](https://dl.acm.org/doi/10.1145/3447548.3467292). KDD 2021. https://github.com/anonymoususer99/T-GAP

[16] (RE-GCN) Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, Xueqi Cheng. ["Temporal Knowledge Graph Reasoning Based on Evolutional Representation Learning"](https://dl.acm.org/doi/10.1145/3404835.3462963). SIGIR 2021. https://github.com/Lee-zix/RE-GCN

[17] (TIE) Jiapeng Wu, Yishi Xu, Yingxue Zhang, Chen Ma, Mark Coates, Jackie Chi Kit Cheung. ["TIE: A Framework for Embedding-based Incremental Temporal Knowledge Graph Completion"](https://dl.acm.org/doi/10.1145/3404835.3462961). SIGIR 2021. https://github.com/JiapengWu/Time-Aware-Incremental-Embedding

[18] (DBKGE) Siyuan Liao, Shangsong Liang, Zaiqiao Meng, Qiang Zhang. ["Learning Dynamic Embeddings for Temporal Knowledge Graphs"](https://dl.acm.org/doi/10.1145/3437963.3441741). WSDM 2021. 

[19] (ST-ConvKB) Jiasheng Zhang, Shuang Liang, Zhiyi Deng, Jie Shao. ["Spatial-Temporal Attention Network for Temporal Knowledge Graph Completion"](https://link.springer.com/chapter/10.1007%2F978-3-030-73194-6_15). DASFAA 2021.

[20] (RETRA) Simon Werner, Achim Rettinger, Lavdim Halilaj, Jürgen Lüttin. ["RETRA: Recurrent Transformers for Learning Temporally Contextualized Knowledge Graph Embeddings"](https://link.springer.com/chapter/10.1007%2F978-3-030-77385-4_25). ESWC 2021. https://github.com/siwer/Retra


### 2020

[1] (TComplEx) Timothée Lacroix, Guillaume Obozinski, Nicolas Usunier. ["Tensor Decompositions for Temporal Knowledge Base Completion"](https://openreview.net/forum?id=rke2P1BFwS). ICLR 2020. https://github.com/facebookresearch/tkbc

[2] (DE-SimplE) Rishab Goel, Seyed Mehran Kazemi, Marcus Brubaker, Pascal Poupart. ["Diachronic Embedding for Temporal Knowledge Graph Completion"](https://aaai.org/ojs/index.php/AAAI/article/view/5815). AAAI 2020. https://github.com/BorealisAI/DE-SimplE

[3] (DArtNet) Sankalp Garg, Navodita Sharma, Woojeong Jin, Xiang Ren. ["Temporal Attribute Prediction via Joint Modeling of Multi-Relational Structure Evolution"](https://www.ijcai.org/Proceedings/2020/386). IJCAI 2020. https://github.com/INK-USC/DArtNet

[4] (DyERNIE) Zhen Han, Peng Chen, Yunpu Ma, Volker Tresp. ["DyERNIE: Dynamic Evolution of Riemannian Manifold Embeddings for Temporal Knowledge Graph Completion"](https://www.aclweb.org/anthology/2020.emnlp-main.593/). EMNLP 2020. 

[5] (RE-NET) Woojeong Jin, Meng Qu, Xisen Jin, Xiang Ren. ["Recurrent Event Network: Autoregressive Structure Inferenceover Temporal Knowledge Graphs"](https://www.aclweb.org/anthology/2020.emnlp-main.541/). EMNLP 2020. https://github.com/INK-USC/RE-Net

[6] (TeMP) Jiapeng Wu, Meng Cao, Jackie Chi Kit Cheung, William L. Hamilton. ["TeMP: Temporal Message Passing for Temporal Knowledge Graph Completion"](https://www.aclweb.org/anthology/2020.emnlp-main.462/). EMNLP 2020. https://github.com/JiapengWu/TeMP

[7] (TIMEPLEX) Prachi Jain, Sushant Rathi, Mausam, Soumen Chakrabarti. ["Temporal Knowledge Base Completion: New Algorithms and Evaluation Protocols"](https://www.aclweb.org/anthology/2020.emnlp-main.305/). EMNLP 2020. https://github.com/dair-iitd/tkbi

[8] (TeRo) Chengjin Xu, Mojtaba Nayyeri, Fouad Alkhoury, Hamed Shariat Yazdi, Jens Lehmann. ["TeRo: A Time-aware Knowledge Graph Embedding via Temporal Rotation"](https://www.aclweb.org/anthology/2020.coling-main.139/). COLING 2020. https://github.com/soledad921/ATISE

[9] (ToKE) Julien Leblay, Melisachew Wudage Chekol, Xin Liu. ["Towards Temporal Knowledge Graph Embeddings with Arbitrary Time Precision"](https://dl.acm.org/doi/10.1145/3340531.3412028). CIKM 2020. https://gitlab.com/jleblay/tokei

[10] (ATiSE) Chenjin Xu, Mojtaba Nayyeri, Fouad Alkhoury, Hamed Shariat Yazdi, Jens Lehmann. ["Temporal Knowledge Graph Completion Based on Time Series Gaussian Embedding"](https://link.springer.com/chapter/10.1007%2F978-3-030-62419-4_37). ISWC 2020. https://github.com/soledad921/ATISE

[11] (TDGNN) Liang Qu, Huaisheng Zhu, Qiqi Duan, Yuhui Shi. ["Continuous-Time Link Prediction via Temporal Dependent Graph Neural Network"](https://dl.acm.org/doi/10.1145/3366423.3380073). WWW 2020. https://github.com/Leo-Q-316/TDGNN

### 2018

[1] (HyTE) Shib Sankar Dasgupta, Swayambhu Nath Ray, Partha Talukdar. ["HyTE: Hyperplane-based Temporally aware Knowledge Graph Embedding"](https://www.aclweb.org/anthology/D18-1225/). EMNLP 2018. https://github.com/malllabiisc/HyTE

[2] (TA-DistMult) Alberto Garcia-Duran, Sebastijan Dumančić, Mathias Niepert. ["Learning Sequence Encoders for Temporal Knowledge Graph Completion"](https://www.aclweb.org/anthology/D18-1516/). EMNLP 2018.


### 2017

[1] (Know-Evolve) Rakshit Trivedi, Hanjun Dai, Yichen Wang, Le Song. ["Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs"](http://proceedings.mlr.press/v70/trivedi17a.html). ICML 2017. 

### 2016

[1] (t-TransE) Tingsong Jiang, Tianyu Liu, Tao Ge, Lei Sha, Sujian Li, Baobao Chang, Zhifang Sui. ["Encoding Temporal Information for Time-Aware Link Prediction"](https://www.aclweb.org/anthology/D16-1260/). EMNLP 2016.


### 2014

[1] (CTPs) Derry Tanti Wijaya, Ndapandula Nakashole, Tom M. Mitchell. ["CTPs: Contextual Temporal Profiles for Time Scoping Facts using State Change Detection"](https://www.aclweb.org/anthology/D14-1207/). EMNLP 2014.
<tr>
	<td>&emsp;<a href=#timestamp-included-tensor-decomposition>2.1 Timestamp-included Tensor Decomposition (TiTD)</a></td>
<td>&ensp;</td>
</tr>
</table>




### [Survey Papers](#content)

1. **Temporal Knowledge Graph Completion: A Survey** ArXiv, 2022. [paper](https://arxiv.org/abs/2201.08236)

    *Borui Cai, Yong Xiang, Longxiang Gao, He Zhang, Yunfeng Li, Jianxin Li.*

## [Approaches](#content)

### [Timestamp-included Tensor Decomposition](#content)

1. **Tensor decomposition-based temporal knowledge graph embedding** ICTAI, 2020. [paper](https://ieeexplore.ieee.org/abstract/document/9288194/)

    *Lin, Lifan and She, Kun*

2. **Tensor decompositions for temporal knowledge base completion** ArXiv, 2020. [paper](https://arxiv.org/abs/2004.04926), [code](Guillaume Obozinski)

    *"Timothee Lacroix*

