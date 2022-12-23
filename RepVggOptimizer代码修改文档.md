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
    # 添加optimizer判别
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
##### 其中[主函数修改]引入了 load_scale_from_pretrained_models, 其作用为从预训练的模型中提取scale因子
    def load_scale_from_pretrained_models(cfg, device):
        weights = cfg.SCALE
        scales = None
        assert weights, "ERROR: NO scales provided to init RepOptimizer"
        ckpt = torch.load(weights, map_location=device)
        scales = extract_scales(ckpt)
        return scales
##### load_scale_from_pretrained_models 用到了另一个函数extract_scales, 其中又包括extract_blocks_into_list
    # 作用：提取预训练scale用于优化器
    def extract_scales(model):
        blocks = []
        extract_blocks_into_list(model['model'],blocks)
        assert blocks != [], 'Error! False extract!'
        scales = []
        for b in blocks:
            if hasattr(b, 'scale_identity'):
                # 根据repvgg，预训练没有bias的就没有identity
                scales.append((b.scale_identity.weight.detach(), b.scale_1x1.weight.detach(), b.scale_conv.weight.detach()))
            else:
                scales.append((b.scale_1x1.weight.detach(), b.scale_conv.weight.detach()))
            print('extract scales: ', scales[-1][-2].mean(), scales[-1][-1].mean())
        return scales
    
    def extract_blocks_into_list(model, blocks):
        for name, module in model.named_children():
            # 不同的存储路径，会显示为不同的类，我将文件放到了layers下的common.py中
            # 此处提取的是预训练backbone中的block
            if str(type(module)) == "<class 'layers.common.LinearAddBlock'>":
                blocks.append(module)
            # 此处提取的是预训练head中的block
            elif name in ["cls.0.conv1","cls.0.block.0","cls.0.block.1","bbox.0.conv1","bbox.0.block.0","bbox.0.block.1",\
                          "kps.0.conv1","kps.0.block.0","kps.0.block.1"]:
                blocks.append(module)
            else:
                # 递归
                extract_blocks_into_list(module, blocks)
##### 此外，[主函数修改]还引入了RepVGGOptimizer,它继承了SGD，并依据《原理》中公式1进行初始化，公式2进行更新
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim.sgd import SGD
    from copy import deepcopy
    
    def get_optimizer_param(cfg, model):
        # 这三行代码作用是,如果batch size 非标准(4，8，16，32)，则对weight decay乘相应因子
        accumulate = max(1,round(64 /cfg.SOLVER.IMS_PER_BATCH))
        WEIGHT_DECAY = deepcopy(cfg.SOLVER.WEIGHT_DECAY)
        WEIGHT_DECAY *= CFG.SOLVER.IMS_PER_BATCH * accumulate / 64
        
        g_bnw, g_w, g_b = [], [], []
        for v in model.modules():
            if hasattr(v,'bias') and isinstance(v.bias, nn.Parameter):
                g_b.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                g_bnw.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g_w.append(weight)
        return [{'params': g_bnw},
                {'params': g_w, 'weight_decay':WEIGHT_DECAY},
                {'params': g_b}]
                
    class RepVGGOptimizer(SGD):
        """ scales is a list, scales[i] is a triple (scale_identity.weight, scale_1x1.weight, scale_conv.weight) or two-tuple if no bias """
        def __init__(self, model, scales, cfg, momentum=0, dampening=0, weight_decay=0, nesterov=True,
                     reinit=True, use_identity_scales_for_reinit=True,cpu_mode=False):
            defaults = dict(lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
            if nesterov and (cfg.SOLVER.MOMENTUM <= 0 or dampening !=0):
                raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            parameters = get_optimizer_param(cfg,model)
            super(SGD,self).__init__(parameters, defaults)
            self.num_layers = len(scales)
            
            blocks = []
            # 上文有此函数
            extract_blocks_into_list(model, blocks)
            convs = [b.conv for b in blocks]
            #  检测预处理提取的所有特殊层数量，等于正式训练时候要处理的特殊层数量
            assert len(scales) == len(convs)
            
            if reinit:
                for m in mdoel.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        gamma_init = m.weight.mean()
                        assert gamma_init == 1.0, '=================== The data is not from scratch ====================='
                # 对训练网络特殊层重新初始化
                self.reinitialize(scales, convs, use_identity_scales_for_reinit)
            # 对训练网络特殊层生成optimizer更新规则的mask
            self.generate_gradient_masks(scales,convs,cpu_mode)
        
        def reinitialize(self, scales_by_idx, conv2x2_by_idx, use_identity_scales):
            for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
                in_channels = conv3x3.in_channels
                out_channels = conv3x3.out_channels
                kernel_1x1 = nn.Conv2d(in_channels, out_channels, 1)
                device = conv3x3.weight.get_device()
                kernel_size = conv3x3.kernel_size
                kernel_1x1.to(device)
                if len(scales) == 2:
                    s0 = scales[0].to(device)
                    s1`= scales[1].to(device)
                    # 这里是我写的拓展，repvgg定死了kernel size是3,如果改成5，1 之类，则需要拓展
                    if kernel_size == 5:
                        pad1 = F.pad(kernel_1x1.weight, [2,2,2,2]).to(device)
                        conv3x3.weight.data = conv3x3.

### 模型backbone修改
