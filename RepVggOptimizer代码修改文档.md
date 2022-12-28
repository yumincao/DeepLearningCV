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
---yaml
<br> +
<br> ---defaults.py
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
--- tools.train_net.py
<br> &nbsp;&nbsp;&nbsp;&nbsp;--添加optimizer判别
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--提取scale
<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--设计优化器
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
##### 其中[添加optimizer判别]引入了 load_scale_from_pretrained_models, 其作用为从预训练的模型中提取scale因子
    def load_scale_from_pretrained_models(cfg, device):
        weights = cfg.SCALE
        scales = None
        assert weights, "ERROR: NO scales provided to init RepOptimizer"
        ckpt = torch.load(weights, map_location=device)
        scales = extract_scales(ckpt)
        return scales
##### load_scale_from_pretrained_models 用到了另一个函数extract_scales, extract_scales又调用了extract_blocks_into_list
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
##### 此外，[添加optimizer判别]还引入了RepVGGOptimizer,它继承了SGD，并依据《原理》中公式1进行初始化，公式2进行更新
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
        
        # 初始化
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
                        conv3x3.weight.data = conv3x3.weight * s1.view(-1,1,1,1) + pad1 * s0.view(-1,1,1,1)
                    elif kernel_size == 3:
                        pad1 = F.pad(kernel_1x1.weight, [1,1,1,1]).to(device)
                        conv3x3.weight.data = conv3x3.weight * s1.view(-1,1,1,1) + pad1 * s0.view(-1,1,1,1)
                    elif kernel_size == 1:
                        pad1 = F.pad(kernel_1x1.weight, [0,0,0,0]).to(device)
                        conv3x3.weight.data = conv3x3.weight * s1.view(-1,1,1,1) + pad1 * s0.view(-1,1,1,1)
                else:
                    assert len(scales) == 3
                    assert in_channels == out_channels
                    s0 = scales[0].to(device)
                    s1 = scales[1].to(device)
                    s2 = scales[2].to(device)
                    identity = torch.from_numpy(np.eye(out_channels, dtype=np.float32).reshape(out_channels,out_channels,1,1)).to(conv3x3.weight.device)
                    identity.to(device)
                    if kernel_size == 3:
                        pad1 = F.pad(kernel_1x1.weight, [1,1,1,1]).to(device)
                        conv3x3.weight.data = conv3x3.weight * s2.view(-1,1,1,1) + pad1 * s1.view(-1,1,1,1)
                        if use_identity_scales:
                            identity_scale_weight = s0
                            pad3 = F.pad(identity*identity_scale_weight.view(-1,1,1,1),[1,1,1,1]).to(device)
                            conv3x3.weight.data += pad3
                        else:
                            pad4 = F.pad(identity,[1,1,1,1]).to(device)
                            conv3x3.weight.data += pad4
                    elif kernel_size == 5:
                        pad1 = F.pad(kernel_1x1.weight, [2,2,2,2]).to(device)
                        conv3x3.weight.data = conv3x3.weight * s2.view(-1,1,1,1) + pad1 * s1.view(-1,1,1,1)
                        if use_identity_scales:
                            identity_scale_weight = s0
                            pad3 = F.pad(identity*identity_scale_weight.view(-1,1,1,1),[2,2,2,2]).to(device)
                            conv3x3.weight.data += pad3
                        else:
                            pad4 = F.pad(identity,[2,2,2,2]).to(device)
                            conv3x3.weight.data += pad4
                    elif kernel_size == 1:
                        pad1 = F.pad(kernel_1x1.weight, [0,0,0,0]).to(device)
                        conv3x3.weight.data = conv3x3.weight * s2.view(-1,1,1,1) + pad1 * s1.view(-1,1,1,1)
                        if use_identity_scales:
                            identity_scale_weight = s0
                            pad3 = F.pad(identity*identity_scale_weight.view(-1,1,1,1),[0,0,0,0]).to(device)
                            conv3x3.weight.data += pad3
                        else:
                            pad4 = F.pad(identity,[0,0,0,0]).to(device)
                            conv3x3.weight.data += pad4
        
        # 更新规则mask
        def generate_gradient_masks(self,scales_by_idx,conv3x3_by_idx,cpu_mode=False):
            self.grad_mask_map = {}
            for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
                para = conv3x3.weight
                if len(scales) == 2:
                    mask = torch.ones_like(para,device=scales[0].device) * (scales[1] ** 2).view(-1,1,1,1)
                    mask[:,:,1:2,1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1, device=scales[0].device) * (scales[0]**2).view(-1,1,1,1)
                else:
                    mask = torch.ones_like(para,device=scales[0].device) * (scales[2] ** 2).view(-1,1,1,1)
                    mask[:,:,1:2,1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1, device=scales[0].device) * (scales[1]**2).view(-1,1,1,1)
                    ids = np.arange(para.shape[1])
                    assert para.shape[1] == para.shape[0]
                    mask[ids, ids, 1:2, 1:2] += 1.0
                if cpu_mode:
                    self.grad_mask_map[para] = mask
                else:
                    self.grad_mask_map[para] = mask.cuda()
                    
        def __setstate__(self, state):
            super(SGD,self).__setstate__(state)
            for group in self.param_groups:
                group.setdefault('nesterov',False)
                
        def step(self, closure=None):
            loss=None
            if closure is not None:
                loss = closure()
            
            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if p in self.grad_mask_map:
                        # 数据在这里乘更新规则mask
                        d_p = p.grad.data * self.grad_mask_map[p]
                    else:
                        d_p = p.grad.data
                        
                    if weight_decay != 0:
                        d_p.add_(weight_Decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                            if nesterov:
                                d_p = d_p.add(momentum, buf)
                            else:
                                d_p = buf
                                
                    p.data.add_(-group['lr'], d_p)
                    
            return loss
        
### 3.模型backbone,head修改
---maskrcnn_benchmark.modeling.backbone.py 注册
<br> +
<br> ---判别用了哪种block（repvgg block所在的库common.py）
<br> +
<br> ---maskrcnn_benchmark.modeling.backbone.resnet.py  修改backbone网络
<br> &nbsp;&nbsp;&nbsp;&nbsp; ---调用LinearAddBlock,RepBlock (repvgg block所在的库common.py)
<br> +
<br> ---FCOS head修改
<br> &nbsp;&nbsp;&nbsp;&nbsp; ---调用LinearAddBlock,RepBlock (repvgg block所在的库common.py)
#### backbone.py 注册
    def make_divisible(x,divisor):
        return math.ceil(x/divisor) * divisor
    
    def get_block(training_mode:str):
        if training_mode == 'hyper_search':
            return LinearAddBlock
        elif training_mode == 'RepVgg':
            return RealVGGBlock
            
    @registry.BACKBONES.register("RepOpt-FPN-YUMIN")
    def build_repvgg_fpn_backbone_ti(cfg):
        # depth_mul = cfg.MODEL.DEPTH_MUL 后续如果想像efficient net一样引入深度因子，宽度因子，则需要使用这里
        # width_mul = cfg.MODEL.WIDTH_MUL
        
        # num_repeats_backbone = [1,6 ,12,18,6]
        # out_channels_backbone=[64,128,256,512,1024]
        # num_repeats = [(max(round(i*depth_mul),1) if i>1 else i) for i in (num_repeats_backbone)]
        # channels_list = [make_divisible(i*width_mul,8) for i in(out_channels_backbone)]
        
        channels_list = [32,64,128,192,384]
        num_repeats = [1,3,3,4,3]
        num_outs=5
        out_channels = 96
        start_level=1
        add_extra_convs=True
        extra_convs_on_inputs=False
        
        block = get_block(cfg.TRAINING_MODE)
        backbone = new_resnet.ResNet(channels_list=channels_list,
                                     num_repeats=num_repeats,
                                     block=block)
        fpn_channels_list = [64,128,192,384] # Hu版本不同
        neck = fpn_module_repopt.FPN(channels_list=fpn_channels_list,
                                        num_outs=num_outs,
                                        out_channels=out_channels,
                                        start_level=start_level,
                                        add_extra_convs=add_extra_convs,
                                        extra_convs_on_inputs=extra_convs_on_inputs)
        backbone.out_channels = out_channels
        neck.fpn_level_num = 5
        head = build_fcos_module(cfg, out_channels, neck.fpn_level_num,block)
        
        return backbone, neck, head
#### [注册]中调用了new_resnet，其结构为：
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    from layers.common import RevVGGBlock, LinearAddBlock, RepBlock
    
    class ResNet(nn.Module):
        def __init__(self, in_channels=3, channels_list=None, num_repeats=None, block=RepVGGBlock):
            super().__init__()
            assert channels_list is not None
            assert num_repeats is not None
            
            self.stem = block(
                            in_channels=in_channels,
                            out_channels=channels_list[0],
                            kernel_size=3,
                            stride=2)
            
            self.ERBlock_2 = nn.Sequential(
                                block(
                                    in_channels=channels_list[0],
                                    out_channels=channels_list[1],
                                    kernel_size=3,
                                    stride=2,
                                    ),
                                RepBlock(
                                    in_channels=channels_list[1],
                                    out_channels=channels_list[1],
                                    n=num_repeats[1],
                                    block=block,
                                    )
                                )
                                
            self.ERBlock_3 = nn.Sequential(
                                block(
                                    in_channels=channels_list[1],
                                    out_channels=channels_list[2],
                                    kernel_size=3,
                                    stride=2,
                                    ),
                                RepBlock(
                                    in_channels=channels_list[2],
                                    out_channels=channels_list[2],
                                    n=num_repeats[2],
                                    block=block,
                                    )
                                )
                                
            self.ERBlock_4 = nn.Sequential(
                                block(
                                    in_channels=channels_list[2],
                                    out_channels=channels_list[3],
                                    kernel_size=3,
                                    stride=2,
                                    ),
                                RepBlock(
                                    in_channels=channels_list[3],
                                    out_channels=channels_list[3],
                                    n=num_repeats[3],
                                    block=block,
                                    )
                                )
                                
            self.ERBlock_5 = nn.Sequential(
                                block(
                                    in_channels=channels_list[3],
                                    out_channels=channels_list[4],
                                    kernel_size=3,
                                    stride=2,
                                    ),
                                RepBlock(
                                    in_channels=channels_list[4],
                                    out_channels=channels_list[4],
                                    n=num_repeats[4],
                                    block=block,
                                    )
                                )
                                
        def forward(self,x):
            outputs=[]
            x = self.stem(x)
            x = self.ERBlock_2(x)
            outputs.append(x)
            x = self.ERBlock_3(x)
            outputs.append(x)
            x = self.ERBlock_4(x)
            outputs.append(x)
            x = self.ERBlock_5(x)
            outputs.append(x)
            
            return tuple(outputs)
#### block所在的库common.py
    在yolov6-layers-common.py, github相关文件里。注意为了量化，修改yolov6中relu为relu6
#### 修改FCOSHead(maskrcnn_benchmark.modeling.rep.fcos.fcos.py)
    # 注销conv部分：for i in range ... add_module...
    # 从backbone的head开始，所有init都添加block， 如【backbone.py 注册】
    self.bbox = nn.Sequential(
                    RepBlock(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        n=self.rep_num,
                        block=block)
                        )
    self.add_module('bbox',self.bbox)
    self.kps = nn.Sequential(
                    RepBlock(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        n=self.rep_num,
                        block=block)
                        )
    self.add_module('kps',self.kps)
    self.cls = nn.Sequential(
                    RepBlock(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        n=self.rep_num,
                        block=block)
                        )
    self.add_module('cls',self.cls)
    
    # 然后修改forward部分
    for l, feature in enumerate(x):
        tmp_Feature = getattr(self,'cls')(feature)
        tmp_feature = self.cls_logits(tmp_feature)
        logits.append(tmp_feature[:,0:self.num_classes,:,:])
        centerness.append(tmp_feature[:,15:16,:,:])
    for l, feature in enumerate(x):
        tmp_feature = getattr(self,'bbox')(feature)
        bbox_pred = self.bbox_pred(tmp_feature)
        if self.norm_reg_targets:
            if self.training:
                bbox..
            else:
                ..
        else:
            ..
    for l, feature in enumerate(x):
        tmp_feature = getattr(self,'kps')(feature)
        kps_pred = self.kps_pred(tmp_feature)
        ..
        ..
    return logits,bbox_reg,kps_reg,centerness
    
### 4.文件存储
    # 注意在checkpoint.py中修改存储方式，存储整个模型而不是state_dict,目的是读取scale
    data["model"] = deepcopy(de_parallel(self.model))
    ..
    torch.save(data,save_file,_use_new_zipfile_serialization=False)
    
    # 而在测试时，需要将模型转换为state_dict:
    terminal:
        python
        import torch
        model = torch.load('model_0020000.pth')
        model = model['model'].state_dict()
        torch.save('format_model_0020000.pth')

