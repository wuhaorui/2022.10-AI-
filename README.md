# **代码使用方式**

## **一.运行环境：**
操作系统：Windows10  <br />
虚拟环境：python3.7.15、torch1.8.0+cu111、torchvision0.9.0+cu111

## 二.文件夹格式
在项目目录下有:<br />
一个vgg_pretrained.py文件、一个参数文件夹params（内含两张模型参数）、一个数据文件夹data <br />
其中params的云盘链接为https://cloud.tsinghua.edu.cn/d/2f85460bf1824f37be71/  <br />
完整代码链接为https://cloud.tsinghua.edu.cn/d/6f8b320c92c947abbff2/
data使用原始文件夹即可

## 三.运行步骤
①在vgg_pretrained.py文件的第86行是加载没有预处理的模型参数
87行加载有预处理的模型参数，选择其中一行运行，注释另一行   <br />
②评估模型：直接运行代码即可  <br />
③训练模型：将第100行注释，并把第189行取消注释（保存参数）
