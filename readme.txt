运行环境：
torch		0.4.1
torchvision	0.4.1
python		3.6
networkx	2.5
scipy		1.5.4

我们在基础GCN网络的基础上建立了ResGCN、DenseGCN模型（见model文件）

data文件夹下为cora、pubmed、citeseer三个数据集

训练模型:在train.py文件内更改模型名称
运行python train.py

对模型进行参数搜索：
python train_search.py