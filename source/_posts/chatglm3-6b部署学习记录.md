---
title: chatglm3-6b部署学习记录
date: 2024-05-20 16:25:25
tags: LLM
categories: 开发教程
---

## 本文作为re0部署ChatGLM3-6B模型过程的记录

目前来说，个人部署大语言模型可以通过阿里云、autodl等平台自带的快速部署功能一键部署，但是本文记录从申请阿里云免费5000cu开始，通过命令行部署的过程。

参考文献[ChatGLM3-6B大模型部署、微调【0到1小白教程】_chatglm3 6b最低部署要求-CSDN博客](https://blog.csdn.net/weixin_44480960/article/details/137092717?csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"137092717"%2C"source"%3A"weixin_44480960"}&fromshare=blogdetail)

#### GPU申请

首先申请这个阿里云的5000计算时产品，有A10和V100两张GUP可选

感谢阿里云的免费计算资源

![image-20240520163240884](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520163240884.png)

稍等一会之后点选免费试用

![image-20240520163848394](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520163848394.png)

点选**第一个**即可，其他两个我暂时还不了解，有需要的可以自己申请用

![image-20240520164011525](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520164011525.png)

点击之后会跳转到控制台，在控制台点选**新建实例**（因为我之前创建过因此存在实例）

![image-20240520164107448](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520164107448.png)

配置选择GPU服务器，其中**A10**和**V100**是可以通过免费计算时抵扣的，二者的区别如下：

A10：显存更大，擅长进行图形计算任务

V100：计算能力更强，显存比A10小，整体多2G内存

其中，A10不关机够用**30**天，V100够用**15**天，虽然是免费资源，**但是还是提醒记得关机~**

（也可以多开资源，但是目前我没尝试过）

![image-20240520164537417](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520164537417.png)

环境选择，我这里是目前官方默认的，建议

- **pytorch2.1.2**
- **tensorflow2.14.0**
- **py310**

以上环境是目前主流pytorch深度学习项目选择的环境，你也可以自己选择喜欢的镜像。

![image-20240520165156759](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520165156759.png)

配置完成后在交互式建模界面可以看到实例，默认应该是开好机的，图示中我已经关机了

![image-20240520165824955](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520165824955.png)

至此，硬件环境安装到此结束。



#### 开始chatglm3-6b部署

```shell
mkdir models
cd models
#创建模型文件夹
apt update
apt install git-lfs
#更新apt和安装git
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
# 克隆chatGLM3-6b大模型

```

阿里云的DSW终端不支持方向键回滚命令...所以还是一条一条复制吧（如果是设置问题希望有人能教我一下怎么把方向键绑定为上一条下一条命令的快捷键）

```shell
mkdir webcodes
cd webcodes
 
git clone https://github.com/THUDM/ChatGLM3.git\
# 下载chatglm3-6b web_demo项目

cd ChatGLM3
#原博主忘了把这一条cd命令加进去

pip install -r requirements.txt
# 安装依赖
```

![image-20240520172513472](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520172513472.png)

有一些版本相关的报错，先跑后面的项目，如果出现依赖问题，再进行修改。

点击阿里云左上角文件标志，进入webcodes/ChatGLM3/basic_demo选择**cli_demo.py**文件

![image-20240520173117842](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520173117842.png)

修改文件路径，用终端cd到对应模型的文件夹下，pwd打出路径进行复制

![image-20240520173551060](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520173551060.png)

最后在cli里替换掉路径就可以了

![image-20240520173656400](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520173656400.png)

之后回到对应目录运行前端

```shell
cd /mnt/workspace/models/webcodes/ChatGLM3/basic_demo/
python cli_demo.py
```

![image-20240520174224228](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520174224228.png)

此时已经正常启动小黑窗进行对话了，然后上面出现的报错是因为一开始输入了**中文逗号**，无法识别为字符串，不清楚是python的问题还是编码的问题。

之后同样是对gradio前端和stream前端路径进行修改，让其读取本地的模型

![image-20240520174620647](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520174620647.png)

修改完成后ctrl+c退出之前运行的小黑窗，运行gradio前端

```shell
python web_demo_gradio.py
```

首次运行的时候我出现了报错ModuleNotFoundError: No module named 'peft'

重新安装一下这个peft

```shell
pip install peft
```

此时重新启动gradio前端就可以正常运行了（**点击阿里云给的本地ip，会自动跳转到域名链接，这一点很不错**）

![image-20240520175229381](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520175229381.png)

进入gradio前端发现无法正常对话，推测是gradio版本的问题，因为终端出现了输入信息，但是没有返回

![image-20240520175739297](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520175739297.png)

```shell
pip show gradio
```

发现我的服务器安装的gradio版本为

**Name: gradio**
**Version: 4.31.4**

换成4.2的版本试过，还是不行，命令行提示4.29版本可以更新，4.29也不行

![image-20240520180624589](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520180624589.png)

```shell
pip install "gradio>=3.38.0,<4.0.0"
```

最终使用了3.50版本，成功解决了部署问题。

![image-20240520181333764](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240520181333764.png)



最后再来试试stream的前端

```shell
python web_demo_streamlit.py
```

报错，显示history没有初始化。

```shell
streamlit run web_demo_streamlit.py
```

使用以上命令可以正常运行，但是打不开网页

最后了解到是因为阿里云的这个GPU服务器不支持IP+端口号访问，streamlit前端只能以后到autodl或者其他GPU云服务器平台再玩了。

至此，chatglm3-6b的部署过程就结束了

过几天尝试chatglm3-6b的微调训练。
