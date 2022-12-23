# 将RepVggOptimizer应用于已有网络
## 原理
<br>repvgg optimizer + ***普通conv*** = 普通optimizer +  ***repvgg***  
在正式训练时，需要将repvgg变种层（用scale代替了batch normalization）提取出来，并用普通的conv代替，初始权重，optimizer更新规则要依据scale计算。
<br>上文***repvgg变种***：一个block的三个分支：(scale_s * conv3x3) + (scale_t * conv1x1) + identity
<br>上文***普通conv***： 应满足
<br>   &nbsp;&nbsp;  1.W<sup>(0)</sup> = s * W<sub>3x3</sub><sup>(0)</sup> + t * W<sub>1x1</sub><sup>(0)</sup>
<br>   &nbsp;&nbsp;  2.W'<sup>(i+1)</sup> $\leftarrow$ W'<sup>(i)</sup> - $\lambda$(1+s<sup>2</sup>+t<sup>2</sup>) $\partial L$ $\div$ $\partial$ W'<sup>(i)</sup>

## 代码实现
### 1. config修改
#### 1.1 在yaml最后添加：
    TRAINING_MODE = ‘**’ 
        # 'hyper_search' ‘RepVgg’ 'Others'
        # 作用：调整预训练/训练
    SCALE = 'path'
        # 作用：为正式训练提供预训练的.pt模型地址 (预训练时SCALE='')
e.g. 
<code>
TRAINIING_MODE: "RepVGG"  
SCALE: "trainiing_dir/hyper_search/model_024000.pth"
</code>
#### 1.2 在yaml定义的MODEL.BACKBONE修改为:
    CONV_BODY = "RepOpt-FPN-YUMIN"
    
#### 1.3 在maskrcnn_benchmark.config写死的defaults.py中添加：
    _C.TRAINING_MODE = "normal"
    _C.SCALE = ""

### 2. main.py(train)修改
#### 2.1 主函数(tools.train_net.py)修改
    添加optimizer判别
    def train(..)
        model = ..
        device = ..
        model.to(device)
        
        if cfg.TRAINING_MODE == 'RepVgg':
            scales = load_scale_from_pretrained_models(cfg,device)
            reinit = True
            optimizer = RepVGGOptimizer(model,scale,cfg,reinit=reinit)
        else:
            optimizer = ..
        scheduler = ..
其中引入了 load_scale_from_pretrained_models, 其作用为从预训练的模型中提取scale因子
    <code>
    def load_scale_from_pretrained_models(cfg, device):
        weights = cfg.SCALE
        scales = None
        assert weights, "ERROR: NO scales provided to init RepOptimizer"
        ckpt = torch.load(weights, map_location=device)
        scales = extract_scales(ckpt)
        return scales
    </code>
&nbsp;&nbsp;&nbsp;&nbsp; load_scale_from_pretrained_models 用到了另一个函数extract_scales在后文^*
此外，还引入了RepVGGOptimizer(SGD),这一函数来自如

### 模型backbone修改
