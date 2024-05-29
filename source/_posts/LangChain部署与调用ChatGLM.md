---
title: LangChain部署与调用ChatGLM
date: 2024-05-29 14:55:56
tags: LLM 
categories: 开发教程
top_img: https://s21.ax1x.com/2024/05/20/pkKyVtU.png
cover: https://s21.ax1x.com/2024/05/20/pkKyVtU.png
---

## 本文作为阿里云部署LangChain的存根记录

### 仓库配置

```shell
pip install langchain==0.0.354
```

库文件有点多，应该要花上五分钟。

![image-20240529150225833](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529150225833-17169796253861.png)

有一些兼容性报错，可以先忽略。

**克隆仓库**

```shell
git clone https://github.com/chatchat-space/Langchain-Chatchat.git
cd Langchain-Chatchat
```

由于之前安装过LLama-Factory和ChatGLM3-6B，我打算先进行配置文件的配置，再看看环境安装有什么问题。

展开配置文件

```shell
python copy_config_example.py
```

然后在项目根目录的config文件夹底下就多出了几个py文件，之后我们在可视化界面进入model_config.py

把auto改成cuda，默认使用gpu计算

![image-20240529151251227](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529151251227.png)

之后下滑，找到我们的GLM3-6B路径，改成本地的

```python
/mnt/workspace/models/chatglm3-6b
```

### 虚拟环境安装

发现没法直接运行，缺环境，但是为了防止langchain的依赖于我本地的两个项目冲突，重装miniconda并且创建新环境。

```shell
cd - 
#/mnt/workspace 回到根目录
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

```

一路回车+yes

```shell
. ~/.bashrc
#/usr/bin/sh: 13: /root/.bashrc: shopt: not found
#/usr/bin/sh: 21: /root/.bashrc: shopt: not found
#(base) \[\e]0;\u@\h: \w\a\]\u@\h:\w$ 
```

阿里云这个AI平台激活conda的base环境命令和别的服务器有些区别，腾讯或者其他的有以下激活方式：

```shell
bash ~/.bashrc
#或者
source ~/.bashrc
```

创建虚拟环境并且激活

```shell
conda create -n langchain python==3.11
conda activate langchain
```

![image-20240529153001210](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529153001210.png)

```shell
cd ../..
cd mnt/workspace/Langchain-Chatchat
pip install transformers==4.37.2
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install -r requirements_webui.txt
```



### 运行模型

我就不信这还不能跑

```shell
python startup.py -a
```

![image-20240529154655651](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529154655651.png)

![image-20240529161128515](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529161128515.png)

![image-20240529161438688](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529161438688.png)



### 添加新模型-Lora

首先到Llama-Factory导出我之前训练的Lora模型，大小大约在15GB左右，因此记得看看自己的硬盘空间

况且，导出建议使用cpu导出，不然会出现阿里云服务器GPU核未初始化的问题

![image-20240529163425409](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529163425409.png)

![image-20240529163339254](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529163339254.png)

之后修改对应的配置文件

![image-20240529163528086](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529163528086.png)

![image-20240529163514672](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529163514672.png)

先剪切掉模型配置的"chatglm3-6b",因为langchain会试图同时加载两个模型，显存会爆。

```shell
. ~/.bashrc
conda activate langchain
cd ../
cd mnt/workspace/Langchain-Chatchat
python startup.py -a
```



![image-20240529164340292](https://jsd.cdn.zzko.cn/gh/Aaaou/Blog-hexo/source/_posts/imgs/image-20240529164340292.png)



### 总结

这次实践完成了langchain的部署和模型的配置，当然，langchain提供的额外的功能和知识库功能我暂时还不清楚用途，慢慢来吧。
