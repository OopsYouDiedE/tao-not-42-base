# TAO-NOT-42 架构数据流与算力分析报告

## 一、 数据流向与张量变换图 (Forward Data Flow)

```mermaid
graph TD
    %% Inputs
    Input[图像帧 t\nShape: 1, 3, 640, 640] --> Stem[Backbone Stem]
    DT[时间步长 dt] --> GRUGate[GRU 状态更新门]
    PrevState[帧 t-1 的 GRU 状态\nShape: 1, 256, 80, 80] --> ConvGRU

    %% Backbone
    subgraph MyYOLOE Backbone (空间特征提取)
        Stem --> Stage2[Layer 0-1: f1, f2 生成]
        Stage2 -->|输出 f1\n1, 64, 320, 320| DepthDec
        Stage2 -->|输出 f2\n1, 128, 160, 160| DepthDec
        Stage2 --> Stage3[Layer 2-16: C3k2, SPPF, C2PSA 融合]
        Stage3 -->|输出原始 P3\n1, 256, 80, 80| ConvGRU
        Stage3 -->|输出原始 P4\n1, 512, 40, 40| YHead
        Stage3 -->|输出原始 P5\n1, 1024, 20, 20| YHead
    end

    %% Temporal Memory
    subgraph 时空融合核心 (ConvGRU Memory)
        ConvGRU((FourierTime\nConvGRUCell))
        ConvGRU -->|原始更新状态| GateCalc[门控混合]
        
        %% Gating logic
        GRUGate -->|16s基频傅里叶展开 + FiLM| GateCalc
        PrevState --> GateCalc
        GateCalc --> NextState[融合后的时空记忆 P3'\nShape: 1, 256, 80, 80]
    end

    %% YOLOE Head (Spatiotemporal Injection)
    NextState -->|截断注入 P3 位置| YHead
    
    subgraph YOLOE Head (时空关联目标检测)
        YHead((YOLOESegment26))
        YHead -->|Proto26| Proto[掩码原型 mask_prototypes\n1, 32, 160, 160]
        YHead -->|cv2/cv3/cv5| DenseOutputs[密集监督输出: 框/类别/系数\n1, 8400, ...]
        YHead -->|one2one 分支| O2OOutputs[跟踪直出 o2o: 框/类别/系数\n1, 8400, ...]
    end

    %% Physical Decoders
    NextState --> DepthDec((Depth\nDecoder))
    NextState --> FlowDec((Flow\nDecoder))
    NextState --> PoseDec((EgoPose\nHead))
    PrevState --> AnomDec((Anomaly\nPredictor))

    subgraph 物理场解码分支 (Physics & Geometry)
        DepthDec --> DepthOut[深度图 depth_pred\n1, 640, 640]
        FlowDec --> FlowOut[光流图 flow_pred\n1, 2, 640, 640]
        PoseDec --> PoseOut[自车位姿 ego_pose\n1, 6]
        AnomDec --> AnomOut[特征异常重构误差\n1, 80, 80]
    end
```

## 二、 640x640 分辨率下的真实算力消耗 (FLOPs)

*以下数据基于全分辨率 640x640 推演，并将正向传播与反向传播的算力严格分开计算：*

| 模块名称 | 功能描述 | 核心张量运算与分辨率 | 预估参数量 (Params) | 正向算力 (Forward FLOPs) | 反向算力 (Backward FLOPs) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOE Backbone** | 提取基础图像空间特征 | 高分辨率滑窗卷积，生成 P3(80x80), P4(40), P5(20) | ~11.5 M | **~26.8 GFLOPs** | **~53.6 GFLOPs** |
| **FourierTimeConvGRU** | 跨帧时空状态记忆融合 | 在 80x80 巨大尺度下计算门控矩阵 (通道 512->256) | ~3.5 M | **~45.3 GFLOPs** | **~90.6 GFLOPs** |
| **YOLOE Head (含双分支)** | 输出 Dense 与 O2O 跟踪框及 Mask | 多尺度 Head 卷积，以及 Proto26 的 160x160 掩码生成 | ~6.5 M | **~15.0 GFLOPs** | **~30.0 GFLOPs** |
| **Depth Decoder** | 从潜空间还原全分辨率深度图 | 连续转置卷积上采样至 640x640 | ~2.8 M | **~5.5 GFLOPs** | **~11.0 GFLOPs** |
| **Flow & Pose Decoders** | 还原 2D 运动与 3D 自车姿态 | 80x80 分辨率下的轻量流解码与池化 | ~1.5 M | **~2.0 GFLOPs** | **~4.0 GFLOPs** |
| **总计 (Total)** | **端到端完整单步训练** | 满载分辨率运行 | **~25.8 M** | **~94.6 GFLOPs** | **~189.2 GFLOPs** |

> [!WARNING]
> **ConvGRU 的高分辨率诅咒**：
> 在 640x640 模式下，单是一个 80x80 分辨率的 ConvGRU 模块，正向算力 (45.3G) 就超越了 16 层的现代卷积主干网络 (26.8G)！由于反向传播需额外消耗两倍算力，一轮训练单张图片总计消耗超过 **283.8 GFLOPs**。在实际训练中，请务必关注 GPU 的显存带宽与算力瓶颈。
