# README

### 配置

单卡GPU：一机一卡3090
多卡GPU：一机两卡3090

### 普通，没有加速框架

#### 训练：

![image](https://github.com/OSH-2024/Team_SwanGeese/assets/144820167/744a0182-2808-4cb6-9d9b-10caefc9aaff)


#### 推理：

![image](https://github.com/OSH-2024/Team_SwanGeese/assets/144820167/435dc30d-0674-405a-ba6e-9cc90f46b1f2)


### 加速框架

#### 训练使用deepspeed框架
![image](https://github.com/OSH-2024/Team_SwanGeese/assets/144820167/6270be18-6e58-43ab-86ea-4dbbbdb8c5f5)


#### 推理使用vllm+deepspeed框架



### 使用Ray

#### Ray+普通
##### 训练：
**一开始我选择通过手动切割数据集的方式来分发数据：**

![image](https://github.com/OSH-2024/Team_SwanGeese/assets/144820167/3a8342f4-8e7a-428c-9111-26260539db87)



训练中显卡使用状况：


![445675112575912697](https://github.com/OSH-2024/Team_SwanGeese/assets/144820167/24958095-23a7-4443-b07e-6bbfc380c19e)



**后来我选择通过显卡自动申请数据请求来分发数据：**

![34017ff3dd95c1d2938e0c9f06b8dca](https://github.com/OSH-2024/Team_SwanGeese/assets/144820167/7d8945eb-c6cd-42e0-a296-9e79b0018995)


训练中显卡使用情况：


![9d099f37edce6b0d52d4acaf430a5d1](https://github.com/OSH-2024/Team_SwanGeese/assets/144820167/c87b8b64-a026-4dcc-9c09-531bd271a0bc)




##### 推理：



#### Ray+加速框架

##### 训练：



##### 推理：

