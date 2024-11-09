
YonYou Network Technology Co., Ltd
- 帮助中国电信国际公司升级BIP系统，帮助该公司实现人力资源在全球分公司共享，并让新版BIP系统与该司常用办公软件（飞书）集成
- 优化SQL语句对PostgreSQL数据的查询，提取数据，使用 Python脚本正则表达式对用户数据进行整理，把旧平台用户数据和新平台用户数据根据格式对齐，在新平台设置新数据格式校验
- 与客户沟通需求，根据不同地点分公司招聘需求，使用UML编写实施方案，在低代码平台上，开发多语言网页，确保网页需要填写的数据符合新需求数据格式，测试产品功能是否符合需求

Hong Kong Great Plan Co., Ltd
- 修改并且部署上线用于香港和中国大陆跨境贸易的电商平台的PC端和移动端
- 基于PHP的ThinkPHP，Laravel和SMARTY框架，，根据OAUTH协议，实现商城平台的微信登陆功能。修改支付接口旧平台代码，根据公司的全新跨境支付需求，根据实时汇率计算价格需求，基于钱方支付平台，分别在实现PC端支付宝支付，微信支付扫码支付，移动端跳转支付功能
- 根据设计团队提供的figama设计稿，修改基于Vue框架的商城网页前端界面



**Implementation and Research of Federated Learning Systems**

- 提出差分隐私算法flip(通过随机化标签数据，达到保护差分隐私的效果) 与flipback算法（随机化上一轮模型中预测值与真实值偏差较大的数据的标签，消减flip 操作对模型精度的影响，并且起到正则项的作用）并使用Paillier算法对梯度加密，使用logistic 回归模型对breast cancer数据集进行纵向联邦学习
- 基于Flower框架，拆分出服务端和客户端，使用GRPC进行通讯，使用同态加密Paillier算法加密梯度信息，使用CNN，ResNet18，LetNet分别对MNIST和CIFAR-10进行横向联邦学习实验
- 基于PyQt-5和QT designer，运用了PyQt-Fluent-Widgets库，设计包含纵向联邦学习和横向联邦学习的系统的GUI

**High-Performance Computing Model and Scheduling Optimization in Cloud Computing**
- 
 -  在单云环境下，使用py4j，gymnasiumm，stable_baselines3，ClousimPlus , Ollama等框架构造出支持多云环境下使用强化学习（DQN，PPO）和LLM（gemma2模型）参与调度优化QOS系统
 - 优化reward计算方式，action space和observation space表示方式把系统从单云环境拓展到多云环境
 - 使用Pegasus框架生成符合项目需求的工作流，确保工作流的结构合理，将Pegasus生成的工作流导入到多云环境的代码中生成Jobs。

  1. 
   3. 在单云环境下CloudsimPlus框架下，==使用Py4J，结合基于Java的CloudsimPlus框架和基于Python的Ollama和Gymnasiumm框架==，集成强化学习算法和LLM生成的调度建议对调度问题进行进一步优化，
   4. 把代码从单云环扩展到到多云环境，集合LLM和强化学习进行调度优化
   5. ==大模型fine tune==
   