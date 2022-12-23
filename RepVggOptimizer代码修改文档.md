# 将RepVggOptimizer应用于已有网络
## 原理
<br>repvgg optimizer + **普通conv** = 普通optimizer +  **repvgg**  
<br>上文**repvgg变种**：一个block的三个分支：(scale_s * conv3x3) + (scale_t * conv1x1) + identity
<br>上文**普通conv**： 应满足
<br>  1.W<sup>(0)</sup> = s * W<sub>3x3</sub><sup>(0)</sup> + t * W<sub>1x1</sub><sup>(0)</sup>
<br>  2.W'<sup>(i+1)</sup> $\leftarrow$ W'<sup>(i)</sup> - $\lambda$(1+s<sup>2</sup>+t<sup>2</sup>) $\frac{a}{b}$ {$\partial$ L} {$\partial$ W'<sup>(i)</sup>}
## 代码实现原理
在正式训练时，需要将repvgg变种层（用scale代替了batch normalization）提取出来，并用普通的conv代替，初始权重，optimizer更新规则要依据scale计算。
### config修改
### main.py(train)修改
### 模型backbone修改
