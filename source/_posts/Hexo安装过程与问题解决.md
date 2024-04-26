---
title: Hexo安装过程与问题解决
date: 2024-04-25 18:39:29
tags:
typora-root-url: Hexo安装过程与问题解决.assets
---





##### 参考文章[零成本搭建一款博客网站(基于Vercel+Hexo完美实现)【保姆级教程】_helo博客建站系统-CSDN博客](https://blog.csdn.net/weixin_52908342/article/details/135173988)

### 环境需求

本地安装Node.js 和 Git，配置好环境变量



### Hexo

win+r管理员权限输入CMD
在命令提示符输入

```shell
npm install -g hexo-cli
```



![image-20240425184750101](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425184750101.png)

找一个准备作为博客仓库的文件夹，地址栏全选输入CMD（也可以直接在cmd里cd到该文件夹），此时我已建站，因此有个myblog

![image-20240425185141220](/image-20240425185141220.png)

```shell
hexo init myblog
```

执行之后正确输出如下

![image-20240425185325821](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425185325821.png)

此时运行

```shell
cd myblog
npm install
hexo server
```

会下载npm（node的包管理器），并且启动基本的hexo博客服务，在地址栏粘贴弹出的地址，进入以下主页，证明框架基本安装完毕



![image-20240425185601131](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425185601131.png)

### 主题配置

这里采用 Github的安知鱼主题

如果使用windows/Linux + git进行部署：

```shell
git clone -b main https://github.com/anzhiyu-c/hexo-theme-anzhiyu.git themes/anzhiyu
```

如果使用git + GitHub + vercel进行部署：

删除文件目录下的theme文件夹（不然在vercel自动化部署时会出现git嵌套导致主题不加载，变成空白网页），之后运行

```shell
npm install hexo-theme-anzhiyu
```

这样就好了，安知鱼主题的官方文档：[进阶配置 | 安知鱼主题官方文档 (anheyu.com)](https://vcl-docs.anheyu.com/advanced/)

如果没有pug和stylus渲染器：

```shell
 npm install hexo-renderer-pug hexo-renderer-stylus --save
```

如果是Linux/macos（防止拉取更新时覆盖掉配置文件）

```shell
cp -rf ./themes/anzhiyu/_config.yml ./_config.anzhiyu.yml
```

最后在项目根目录找到网页配置文件，将theme后的值修改成anzhiyu,作者和博客地址什么的也是在这个配置文件

![image-20240425191140144](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425191140144.png)

设置完毕后重新唤醒网页，我这个稍微改了点东西，一般正常的部署完是作者设置的hello world

```shell
hexo server
```

![image-20240425191627146](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425191627146.png)

![image-20240425191755468](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425191755468.png)

至此，正常的项目落地教程结束，接下来开始GitHub+vercel部署教程：

### Github

首先创建一个空仓库，在主页正中将仓库地址复制下来（如果你创建的时候勾选了添加readme，那地址就在绿色code按钮里）

### Git

```shell
git config --global user.email "你的邮箱地址" //绑定邮箱

git config --global user.name "你的用户名"	//绑定用户

git init //初始化一个git项目

git add . //将本地仓库加载到git缓存中

git commit -m "描述" // 将文件的更改保存到本地代码仓库中

git remote add origin 仓库地址 //链接在线仓库

git push -u origin master 	//将本地仓库更新到线上

git pull origin master //将线上仓库更新到本地（若本地仓库与线上存在额外文件，需要先更新本地，当然，也可以在push里使用force强行替换）
```

执行完之后大致仓库如下

![image-20240425192752256](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425192752256.png)

### Vercel

首先用你的GitHub注册vercel~GitHub网页**右上角**有他们的地址

[vercel/vercel: Develop. Preview. Ship. (github.com)](https://github.com/vercel/vercel)

![image-20240425193221563](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425193221563.png)

注册完成登录见如下图片，点击**右边**那个**Add New**-Project

![](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425192903088.png)

选定咱创建的仓库

![image-20240425193853050](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425193853050.png)

修改名称（只会影响vercel随机分配的域名）

![image-20240425194025109](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimgimage-20240425194025109.png)

## 此时你大概率会报错

Error: Command "npm run build" exited with 126 vercel........

别慌，解决方法给你找好了（快速部署直接跳到下一个代码块，底下对很多人来说是废话）

##### 先说结论

这个报错一般是Linux系统权限不足或者缺少依赖，但是我们在本地Windows/Linux上能跑呀，因此只能是Linux权限不足了，众所周知Linux系统授权执行一般有两种方法：root账户执行和sudo执行，由于vercel这个项目是自动化部署的，没有提供给用户的终端面板，因此我们只能想办法让其以sudo的形式执行。

因此，打开配置文件package.json，将build一行修改为：

```json
chmod -R 777 node_modules/ && hexo generate
```

代表赋予所有用户（拥有者、所属组、其他用户）读取、写入和执行的权限。~~这段授权代码让我想起了实习时候搞嵌入式开发的悲惨历程~~

![image-20240425194631335](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425194631335.png)

之后执行以下内容将更新加载到仓库

```shell
git add . //将本地仓库加载到git缓存中

git commit -m "描述" // 将文件的更改保存到本地代码仓库中

git push -u origin master 	//将本地仓库更新到线上

```



此时，删除原来的vercel部署的项目（不是很确定，因此推荐在setting的常规设置滑到最底下删除旧的部署，因为我必须要删除重新部署，不然不能应用上游更改），重新执行前面的部署，等待彩条喷出，网站即可正确部署。

部署完毕会给咱一个域名，可惜的是。vercel的域名已经被污染得差不多了。

此时需要代理工具，访问项目给咱的域名，于是就能够浏览自己的博客了~。

### 自定义域名

如果咱自己有域名的话（没有的话不行哦，得自己购买并且实名），按照以下操作

![image-20240425195944460](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425195944460.png)

一般有A和CNAME两种自定义域名方式

### A

即Apex，顶级域名，输入框3之后会有一个框内提示域名未初始化，并且给出一个IP。

将这个IP填入咱自己的域名提供商的解析ip就可以了、

### CNAME

这个是二级域名，自定义域名方式是域名转域名，输入框3后的值Value就是域名提供商页面的“目标值”，填好即可。



最后完成后显示如图（虽然不知道打码的意义是什么但是按照惯例域名还是码了）

![image-20240425200457461](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240425200457461.png)

最后就可以通过域名访问咱自己的博客了~

有空的话下一篇写怎么更新这个博客~
