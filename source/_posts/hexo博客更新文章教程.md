---
title: hexo博客更新文章教程
date: 2024-05-19 15:13:41
tags: 
categories: 开发教程
---

## 本文将系统性介绍如何通过hexo博客写文章上传

如果阅读本文前未部署个人博客请结合[Hexo安装过程与问题解决 | Hexo-Aou (aou123.xyz)](https://blog.aou123.xyz/2024/04/25/Hexo安装过程与问题解决/)使用



#### 进入hexo

打开**git cmd**，先输入对应的盘符（比如我本地仓库在D盘，需要先输入D:回车，默认C盘不需要操作）

移动到博客本地仓库，再使用hexo new命令新建一个文章

```shell
cd 本地仓库地址
hexo new "你的文章名字"
```



![image-20240519152223867](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240519152223867.png)

之后在\source\\_posts文件夹下可以看见新建的md文件

![image-20240519152735054](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240519152735054.png)



#### 编辑文本

使用md编辑器打开文件，我使用的是**Typora**，大家可以使用自己喜欢的编辑器

文件内自动生成了一个hexo的标题，冒号后面的内容都是可以进行修改的，其中tags需要自己去阅读hexo或者anzhiyu主题的文档进行添加功能。

![image-20240519153758654](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240519153758654.png)

其中，有一个小bug，当第一个输入的正文字符为中文时，会自动写到tags：后面，因此我的解决方案是先如图输入一个数字或者西文字符，写完再第一行再把它删除。



#### 配置图床

**图床**简单来说就是用于存储图片的服务器，由于博客部署在vercel上，无法直接通过相对路径索引的形式加载图片，这时就需要用到图床。

**如果博客部署在个人服务器上**，**那么图床服务可以不按照以下教程进行**（我没有能够作为图床的个人服务器，不清楚服务器环境下hexo能否直接从本地加载图片）



首先安装**PicGo**，这是一个开源的图床软件，我安装的版本是2.4.0

[Releases · Molunerfinn/PicGo (github.com)](https://github.com/Molunerfinn/PicGo/releases)

安装完成后，进入图床设置 本文使用**GitHub**作为图床

仓库名为GitHub主页显示的：

![image-20240519160055399](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240519160055399.png)

分支名为你自己的分支名，如果完全按照我的教程来，应该是master，使用默认分支名的话，一般情况是main，可以在GitHub仓库的顶部分支看到。

token去GitHub-setting-develop setting里申请，具体教程在百度上有很清晰的图文

![image-20240519160816009](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240519160816009.png)

储存路径可以自己新建文件夹但是注意，picgo的储存路径最后一项不是文件夹，是图片头的名字（即会给图片加一个前缀）

![image-20240519155601082](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240519155601082.png)

CDN加速用的是https://jsdelivr.codeqihan.com/

加上gh/GitHub用户名/仓库名

是启涵大佬的镜像[自制的一个jsDelivr镜像分享 - 启涵的博客 (codeqihan.com)](https://www.codeqihan.com/post/zi-zhi-de-yi-ge-jsdelivr-jing-xiang-fen-xiang/)



配置完成后，在typora偏好设置中配置picgo图床

![image-20240519234608363](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240519234608363.png)

写完博客后直接在格式-图像-上传所有本地图片就ok了

![image-20240519234535806](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240519234535806.png)

如果上传失败，在保证配置正确的情况下，优先考虑是不是开了代理，包括但不限于**steam++**，科学上网等

图片传完记得去git推送一下更新。

```shell
git add .
git commit -m "博客更新"
git pull
git push
```



本文至此告一段落。
