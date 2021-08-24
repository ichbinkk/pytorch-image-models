# KK的paper笔记(持续更新)


##1 基于注意力的特征融合
* Attentional Feature Fusion [[paper](https://arxiv.org/abs/2009.14082)][[code](https://github.com/YimianDai/open-aff)]
  * 
  * 

##2 多模态融合
* Memory Fusion Network for Multi-View Sequential Learning，AAAI 2018 [[paper](https://arxiv.org/pdf/1802.00927.pdf)][[code](https://github.com/pliang279/MFN)]
  * Memory Fusion Network（MFN）就是一种使用“Delta-memory attention”和“Multi-View Gated Memory”来同时捕捉时序上和模态间的交互，以得到更好的多视图融合。模型图如下，用memory的目的是能保存上一时刻的多模态交互信息，gated过滤，Attention分配权重。

* Multi-Interactive MemoryNetwork
  * 为了捕获多模态间和单模态内的交互信息，模型又使用了Multi-interactive attention机制。即Textual和Visual在多跳的时候会相互通过Attention来融合信息。

* Adversarial Multimodal Representation Learning for Click-Through Rate Prediction [[paper](https://dl.acm.org/doi/pdf/10.1145/3366423.3380163)]
  * 通过不同的考虑模态特异性和模态不变特征来考虑模态的非定性和冗余性。
  * 在多模态融合（普通的Attention融合，即图中的MAF）旁边加上一个双判别器对抗网络（即图中的DDMA），即分别捕捉动态共性，和不变性。
  * 双判别器是为了挖掘：1，各种模式共同特征的潜在模式（第一个D 识别可能来自共同潜在子空间的模态不变特征，跨越多种模式并强调已识别的模态不变特征，以进一步混淆第二个鉴别器）。
  2，并推动各种模式之间的知识转让（第二个D 在模式之间学习跨多个模式的共同潜在子空间）。
    
* Cross-modality Person re-identification with Shared-Specific Feature Transfer
  * 提出了一种新的跨模态共享特征转移算法(cm-SSFT)
  
* Feature Projection for Improved Text Classification [[paper](https://aclanthology.org/2020.acl-main.726.pdf)]
  * 共性和个性的文章还有这一篇，ACL 2020。基础思路是用特征投影来改善文本分类。
  
* Multi-modal Circulant Fusion for Video-to-Language and Backward [[paper](https://www.ijcai.org/proceedings/2018/0143.pdf)]
  * 同时使用vector和matrix的融合方式，通过circulant matrix讲vector的每一行都平移一个元素得到matrix，这样以探索不同模态向量的所有可能交互。
  
* Learning Deep Multimodal Feature Representation with Asymmetric Multi-layer Fusion [[paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413621)]
  * 私有化BN即可统一多模态的表示。
  * 双向不对称fusion
  
* Adaptive Multimodal Fusion for Facial Action Units Recognition [[paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413538)]
  * 自动从模态中选取最合适的模态特征
  * 同时从三个模态的特征中进行采样。单个模态得到特征后橫着拼接成矩阵，然后通过采样在每维上自动选取最合适的特征，并且可以通过多次采样得到更丰富的表示。
  * 此时采样之后变成离散的了，无法进行梯度传播，所以作者借用了VAE里面重参数技巧，用Gumbel Softmax来解决了。
  
* Attention Bottlenecks for Multimodal Fusion [[paper](https://arxiv.org/pdf/2107.00135.pdf)]
  * 在两个Transformer间使用一个shared token，从而使这个token成为不同模态的通信bottleneck以节省计算注意力的代价
  
*
  *
  
*
  *
  
*
  *

