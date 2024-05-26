---
title: Hexo主题应用-代码框更改
date: 2024-05-27 00:40:24
tags: [Hexo,git]
categories: 开发教程
top_img: https://s21.ax1x.com/2024/05/20/pkKysN8.png
cover: https://s21.ax1x.com/2024/05/20/pkKysN8.png
---

### 今天把博客的代码框换成了我个人更喜欢的类型

使用的**prismjs**

比起原来的**highlight**，我更喜欢它的**抬头显示 **就是那个巨大的**代码类型和开头的三个点**

以下是二者成品的区别，前者为**prismjs**，后者为**prismjs**

![image-20240527004300870](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240527004300870.png)

![image-20240527004414551](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240527004414551.png)



### 检查hexo版本

首先打开Git Bash（别问为什么不用CMD了，两个差不多，主要我之前出了个小乌龙，跑hexo用不了bash）

~~**其实就是我之前本地有个node文本和node.exe重名导致bash以为那个文本文件是可执行文件**~~

bash的命令偏向Linux，缺点是默认的cv没有快捷键

进入我们的博客根目录（**记得/开头**）

```shell
cd /d/hexo/myblog
```

然后检查hexo版本，在本文发出之后安装的版本一般都>=7.2

```shell
hexo -v
```



![image-20240527005111914](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240527005111914.png)



为什么要检查hexo版本：

因为在hexo官方文档中，7.0以后的hexo对**highlight**和**prismjs**的参数支持有区别。



### 高亮代码框

打开hexo根目录的配置文件

找到**highlight**和**prismjs**对应的控件参数，它是在安装时默认生成的。

![image-20240527005457369](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240527005457369.png)



如果版本为**7.0及以上**

```yaml
syntax_highlighter: prismjs #修改为需要更换的代码渲染器类型，我选择的是prismjs，默认highlight.js
highlight:
  line_number: true
  auto_detect: false
  tab_replace: ''
  wrap: true
  hljs: false
prismjs:
  enable: true #在启用的渲染器之后添加enable参数，这是主题需要读取的
  preprocess: true
  line_number: true
  line_threshold: 0
  tab_replace: ""
```

**提醒**：如果你需要用highlight.js，那就在highlight下方添加enable: true参数。

这个参数hexo文档中显示7.0及以后版本已经移除，但是安知鱼主题以及其他类似的采用了这两个代码渲染器的主题仍然需要读取这个参数才能调用主题的额外参数

![image-20240527010039409](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240527010039409.png)

低于7.0的版本：

```yaml
# _config.yml
highlight:
  enable: false
prismjs:
  enable: true
  preprocess: true
  line_number: true
  line_threshold: 0
  tab_replace: ''
```



可以发现，早期版本和当前较新的版本存在一定的区别。

早期版本需要全部设置enable参数，而新版本的原生hexo则不需要设置enable参数，但是主题配置文档需要接收enable参数，因此，需要将启用的代码块渲染器设置enable: true

以下是我使用的主题和hexo代码高亮二者的官方文档，需要进一步钻研的小伙伴可以看看。

[代码高亮 | Hexo](https://hexo.io/zh-cn/docs/syntax-highlight)

[基础配置 | 安知鱼主题官方文档 (anheyu.com)](https://docs.anheyu.com/global/base.html)

以上为Hexo原文档和安知鱼主题官方文档



### 成果展示

在安知鱼主题配置文件添加以下参数

分别是：代码风格，复制功能，显示语言，自动折叠，代码限高和自动换行

在上面的安知鱼文档内有详细说明，当然，以下配置可直接运行。

```yaml
highlight_theme: mac #  darker / pale night / light / ocean / mac / mac light / false
highlight_copy: true # copy button
highlight_lang: true # show the code language
highlight_shrink: false # true: shrink the code blocks / false: expand the code blocks | none: expand code blocks and hide the button
highlight_height_limit: 330 # unit: px
code_word_wrap: false
```

运行本地博客

```shell
hexo server
```

![image-20240527011030200](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240527011030200.png)

以上，可以看到我们的高级代码块了！！！



### 总结

这个折磨了我好长时间，当我写了一晚上的链表题目和题解之后，突发奇想，想要把代码块改成这种带复制功能并且圆角好看带高亮的形式时，我便去网上找各种文档和教程。

之后就把重心放在了官方文档和主题官方文档是看。

最终还是靠自己百炼成钢了，虽然不是什么很高技术力的东西，但是我调配置文件来来回回试了好几十次，从八点多开始，各种奇怪的代码块风格一致纠缠到了十二点多，最后终于发现，问题出在主题文件和hexo文件参数不对等上。

改完参数跑出来的那一刻真的太开心了，博客也差不多焕然一新了。

不过随着版本的更替，这个问题总有一天会得到改善的，咱们计算机还真是需要终身学习啊。

对了，今天还学了一串新的git命令

```shell
git log #查看各版本日志
git reset --hard HEAD^ #回退到最新版本
git reset --hard hash #回退到对应hash值版本
```



**云中谁寄锦书来，雁字回时，月满西楼**。

$愿各位都能成为生活的高手。$

**那么各位，晚安。**

