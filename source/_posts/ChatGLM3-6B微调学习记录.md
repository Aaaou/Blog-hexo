---
title: ChatGLM3-6B微调学习记录
date: 2024-05-23 19:35:24
tags: LLM 
categories: 开发教程
top_img: https://s21.ax1x.com/2024/05/20/pkKyVtU.png
cover: https://s21.ax1x.com/2024/05/20/pkKyVtU.png
---

### ChatGLM3-6B微调学习记录

前文学习了ChatGLM3-6B模型在阿里云DSW服务器上的部署，本文对微调此模型的过程做一个记录。

如果需要学习部署可以移步前文[chatglm3-6b部署学习记录 | Hexo-Aou (aou123.xyz)](https://blog.aou123.xyz/2024/05/20/chatglm3-6b部署学习记录/)



#### 微调工具：LlaMa-Factory

[hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs (github.com)](https://github.com/hiyouga/LLaMA-Factory)

该工具提供一个清晰明确的gradio前端，方便用户进行交互式调参

![image-20240523194106731](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523194106731.png)

首先克隆该项目

```shell
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
#安装依赖
pip install -r requirements.txt
pip install transformers_stream_generator bitsandbytes tiktoken auto-gptq optimum autoawq
pip install --upgrade tensorflow
pip install -e .[torch,metrics]
#运行
#以下版本已停用
#CUDA_VISIBLE_DEVICES=0 USE_MODELSCOPE_HUB=1 python src/train_web.py
#现在需要进入src目录运行webui.py文件，推测是更新版本后统一了命名
CUDA_VISIBLE_DEVICES=0 USE_MODELSCOPE_HUB=1 python src/webui.py
```

**一个小报错**：`Cannot open data/dataset_info.json due to [Errno 2] No such file or directory: 'data/dataset_info.json'.`

之前为了图方便是cd到了src目录看了下webui改成啥名字了，导致从src下直接python了webui.py

用上面修改后的shell命令就可以正常读取到自带的数据集。

**修改前**

![image-20240523195153331](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523195153331.png)

**修改后**

![image-20240523201403309](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523201403309.png)

点击URL，即可进入到LLama的微调界面，点选中文。

![image-20240523195348022](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523195348022.png)

之后把模型选中ChatGLM3-6B-Chat，模型路径填入之前复制到前端界面的模型路径

```shell
/mnt/workspace/models/chatglm3-6b
```



#### 模型微调

微调方法选择lora，lora是目前比较主流的微调模型训练方案，在选择模型的时候加上lora模型的权重，生成的时候就会调用对应的lora模型。

在计算资源充足的情况下也可以使用full，全量微调，个人感觉应该是会直接在原模型上进行训练，生成的模型应该也是完整的，不需要额外添加权重参数的模型，但是应该会需要非常巨大的计算资源。

![image-20240523195941533](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523195941533.png)

作为尝试，我们先使用来自lalama自带的数据集

`alpaca_zh_demo.json`

这个数据集内容大致是用GPT翻译过来的中文问答。

![image-20240523211307451](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523211307451.png)



第一次运行在加载模型中途出现了类似`CUDA error: device kernel image is invalid`之类的报错。

**解决方式**：~~重启服务器。~~

多次重复操作即可，这个错误是阿里云的GPU没反应过来，多次申请调用GPU就可以了。



使用默认参数，先跑三轮试试。（我个人习惯先使用预定义好的参数进行一次尝试）

况且这个数据集总的来说还算比较庞大，网络上大多数人的反馈是在使用精简数据集时50轮以上才有比较明显的效果。

![image-20240523211423933](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523211423933.png)

不得不说LLama的前端真的很人性化，做了损失函数折线图和训练轮数的进度条，不过我觉得输出框应该限制一下长度（输出数据太多了），虽然我也不会写，但是在sovits的整合包里看到过有人能写出来。

![image-20240523212233490](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523212233490.png)

训练完成之后刷新一下适配器

![image-20240523213212442](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523213212442.png)

以下是使用了训练的**Lora模型**的效果

![image-20240523213437417](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523213437417.png)

以下是**原模型**的效果

![image-20240523213609821](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523213609821.png)

可以看到，Lora模型很明显是起作用的，就像Stable-Diffusion的Lora模型可以改变画风一样，LLM的Lora也可以起到包括但不限于改变自我认知，拓宽知识面，语言专业化等效果。



#### 自定义模型

我们下载之前所提到的dataset配置文件。

![image-20240523214052614](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523214052614.png)

可以看到，该文件的格式为

```json
{
  "前端显示的数据集名": {
    "file_name": "所加载的数据集.json"
  },
```

也即是说，我们自定义数据集的话，准备一份如下图的数据集

![image-20240523214241120](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523214241120.png)

![image-20240523214255418](https://jsdelivr.codeqihan.com/gh/Aaaou/Blog-hexo/source/_posts/imgs/imgimage-20240523214255418.png)

```json
[
  {
    "instruction": "提问",
    "input": "",
    "output": "回答"
  },
  ......
 ]
```

也就是以上json格式的文件，再将文件名填入dataset配置文件即可加载自定义数据集。



至此，ChatGML3-6B的微调学习告一段落。

#### 总结：

每次学习这种项目，挫败感最强的一般不是难以看懂的各种神经网络框架或者文件处理代码，最让人心烦的往往就是运行环境，比如这次的这个cuda图像核的报错，搜索了半天都没个结果，搜到的第一个结论是降低pytorch版本，但是项目推荐的版本就是2.2，按教程改成1.2甚至会连cuda都不匹配。

在尝试重启之后短暂缓解了问题，在多次尝试之后发现这只是阿里云GPU服务器的一个小bug，多点几次其实就解决问题了。

**~~只能说跑模型就和运维一样啊，越老越吃香。~~**

