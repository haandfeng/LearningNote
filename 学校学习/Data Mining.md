论文1
使用DDIM（和DDPM不同点在于，去噪的时候加入条件）
改进：加入了下面的语义编码（原本的语义编码是时间戳和情绪标签）
	CNN->语义编码
	VAE->面部特征
原文：
	这种设计通过利用来自不同来源（CNN或VAE）的编码，强化了模型的能力，以更精确地生成与目标情绪对应且保持输入个体特征的面部表情。全连接层的使用是为了确保从这些复杂的编码中提取出对生成任务有用的、高度凝练的特征。通过将这些特征与情绪标签和时间戳嵌入结合，我们可以精确控制生成过程，以反映期望的表情变化，同时保持面部的识别一致性。



初步思路 使用斯坦福大学文章提到的技术，把数据集的每一张图都做出7种不同的表情，然后把他们放到数据集里面训练

[数据集合](http://www.whdeng.cn/RAF/model1.html#dataset)

[SOTA链接](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013)
论文
Facial Expression Recognition using Residual Masking Network
中文详讲链接 [# 论文阅读](https://blog.csdn.net/sinat_39544237/article/details/120203781)




[扩散1](https://zhuanlan.zhihu.com/p/625386246)
[扩散2](https://zhuanlan.zhihu.com/p/591063051)