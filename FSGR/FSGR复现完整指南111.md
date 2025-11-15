# FSGR模型复现完整指南及PMA-Net融合方案

## 目录
1. [环境配置](#环境配置)
2. [数据和资源准备](#数据和资源准备)
3. [FSGR模型复现步骤](#fsgr模型复现步骤)
4. [PMA-Net解码器融合方案](#pma-net解码器融合方案)
5. [训练和评估](#训练和评估)

---

## 环境配置

### 1. 基础环境
```bash
# 已完成:clone代码库
cd FSGR

# 已完成:创建conda环境
conda env create -f environment.yml
conda activate m2release

# 下载spacy数据
python -m spacy download en_core_web_sm
```

---

## 数据和资源准备

### 1. COCO数据集下载

#### 1.1 图像数据
```bash
# 创建数据目录
mkdir -p datasets/coco/images
cd datasets/coco/images

# 下载训练集图像(13GB)
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip

# 下载验证集图像(6GB)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

# 下载测试集图像(可选,用于在线评测)
wget http://images.cocodataset.org/zips/test2014.zip
unzip test2014.zip
```

#### 1.2 标注文件
```bash
cd ../
mkdir annotations
cd annotations

# 下载标注文件
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

# 下载Karpathy split (用于标准评测)
# 这个split是标准的train/val/test划分
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip
```

#### 1.3 M2 Transformer标注格式
FSGR使用M2 Transformer的标注格式,需要准备以下文件:
```bash
# 在项目根目录创建m2_annotations文件夹
cd ../../..
mkdir m2_annotations
```

从RSTNet仓库获取预处理的标注:
```bash
# 下载M2格式标注
git clone https://github.com/zhangxuying1004/RSTNet.git temp_rstnet
cp -r temp_rstnet/m2_annotations/* m2_annotations/
rm -rf temp_rstnet
```

或者手动准备标注文件,需要包含:
- `captions_train2014.json`
- `captions_val2014.json`
- `coco_train_ids.npy`
- `coco_dev_ids.npy`
- `coco_test_ids.npy`

### 2. CLIP预训练模型

#### 2.1 下载CLIP ViT-B/16模型
```bash
# 创建缓存目录
mkdir -p .cache/clip

# 下载CLIP ViT-B/16权重
cd .cache/clip
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

cd ../..
```

### 3. MaskCLIP语义标签提取

#### 3.1 克隆MaskCLIP仓库
```bash
cd ..
git clone https://github.com/chongzhou96/MaskCLIP.git
cd MaskCLIP
```

#### 3.2 提取对象-语义码本(object-semantic codebook)
```bash
# 安装MaskCLIP依赖
pip install -r requirements.txt

# 提取CLIP文本嵌入作为语义码本
# 这个脚本需要根据MaskCLIP的代码自定义
python extract_text_embeddings.py \
    --clip-model ViT-B/16 \
    --output-path pretrain/ram_ViT16_clip_text.pth
```

**自定义提取脚本 `extract_text_embeddings.py`:**
```python
import torch
import clip
import argparse
import numpy as np

def extract_clip_text_embeddings(clip_model_name, output_path):
    """
    提取CLIP文本嵌入作为语义码本
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model_name, device=device)
    
    # COCO类别(80类 + 背景)
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'zhandbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # 为每个类别创建提示模板
    text_prompts = [f"a photo of a {cls}" for cls in coco_classes]
    
    # 文本编码
    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 保存
    torch.save({
        'text_features': text_features.cpu(),
        'classes': coco_classes,
        'prompts': text_prompts
    }, output_path)
    
    print(f"Text embeddings saved to {output_path}")
    print(f"Shape: {text_features.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-model', default='ViT-B/16', type=str)
    parser.add_argument('--output-path', default='pretrain/ram_ViT16_clip_text.pth', type=str)
    args = parser.parse_args()
    
    extract_clip_text_embeddings(args.clip_model, args.output_path)
```

#### 3.3 提取patch-level语义标签(可选)
如果使用 `--return_index` 参数,需要准备patch-level的语义标签:

```bash
# 使用MaskCLIP提取密集语义标签
python extract_patch_labels.py \
    --coco-root ../datasets/coco/images \
    --clip-model ViT-B/16 \
    --output-path patch_labels.hdf5
```

**自定义提取脚本 `extract_patch_labels.py`:**
```python
import torch
import clip
import h5py
import os
from PIL import Image
from tqdm import tqdm
import argparse

def extract_patch_labels(coco_root, clip_model_name, output_path):
    """
    为COCO图像提取patch-level的语义标签
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model_name, device=device)
    
    # 创建HDF5文件
    h5_file = h5py.File(output_path, 'w')
    
    # 遍历train2014和val2014
    for split in ['train2014', 'val2014']:
        img_dir = os.path.join(coco_root, split)
        img_files = sorted(os.listdir(img_dir))
        
        print(f"Processing {split}...")
        for img_file in tqdm(img_files):
            img_path = os.path.join(img_dir, img_file)
            img_id = int(img_file.split('_')[-1].split('.')[0])
            
            # 加载图像
            image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
            
            # 提取grid特征(假设使用7x7=49个patches)
            with torch.no_grad():
                # 这里需要根据CLIP的具体实现提取中间层特征
                # 简化版本:提取全局特征
                image_features = model.encode_image(image)
                
            # 保存
            h5_file.create_dataset(f'{img_id}_labels', 
                                  data=image_features.cpu().numpy())
    
    h5_file.close()
    print(f"Patch labels saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-root', required=True, type=str)
    parser.add_argument('--clip-model', default='ViT-B/16', type=str)
    parser.add_argument('--output-path', default='patch_labels.hdf5', type=str)
    args = parser.parse_args()
    
    extract_patch_labels(args.coco_root, args.clip_model, args.output_path)
```

### 4. 预训练语言模型(可选)

如果要使用language model预训练,下载RSTNet的预训练权重:
```bash
cd ../FSGR
mkdir Pre-trained_Models
cd Pre-trained_Models

# 从RSTNet获取语言模型权重(如果可用)
# 这一步是可选的,论文中先训练language model再训练transformer
wget [RSTNet语言模型权重URL]  # 需要查找具体URL或自己训练
```

---

## FSGR模型复现步骤

### 阶段1: 训练语言模型(可选但推荐)

```bash
cd FSGR

# 训练语言模型
python train_language.py \
    --exp_name fsgr_language \
    --batch_size 50 \
    --workers 4 \
    --features_path datasets/coco/images \
    --annotation_folder m2_annotations
```

### 阶段2: 训练完整Transformer模型

#### 2.1 不使用text embedding(基础版本)
```bash
python train_transformer.py \
    --exp_name fsgr_baseline \
    --batch_size 100 \
    --workers 4 \
    --features_path datasets/coco/images \
    --annotation_folder m2_annotations \
    --pre_vs_path .cache/clip/ViT-B-16.pt \
    --pre_name ViT-B/16
```

#### 2.2 使用完整FSGR(包含text embedding)
```bash
python train_transformer.py \
    --text \
    --exp_name fsgr_full \
    --batch_size 100 \
    --workers 4 \
    --return_index \
    --features_path datasets/coco/images \
    --annotation_folder m2_annotations \
    --text_embed_path ../MaskCLIP/pretrain/ram_ViT16_clip_text.pth \
    --labels_path ../MaskCLIP/patch_labels.hdf5 \
    --pre_vs_path .cache/clip/ViT-B-16.pt \
    --pre_name ViT-B/16 \
    --head 8 \
    --adapter_b 6 \
    --adapter_e 11
```

**关键参数说明:**
- `--text`: 启用aligned cross-modal interaction
- `--return_index`: 启用patch-level语义标签
- `--adapter_b 6 --adapter_e 11`: adapter层的起始和结束位置
- `--head 8`: 注意力头数量
- `--batch_size 100`: 根据GPU内存调整

### 阶段3: 模型评估

#### 3.1 离线评估(Karpathy test split)
```bash
python test_transformer.py \
    --model_path save_models/fsgr_full_best.pth \
    --vocab_path vocab.pkl \
    --text \
    --features_path datasets/coco/images \
    --annotation_folder m2_annotations \
    --text_embed_path ../MaskCLIP/pretrain/ram_ViT16_clip_text.pth \
    --pre_vs_path .cache/clip/ViT-B-16.pt \
    --pre_name ViT-B/16 \
    --batch_size 10 \
    --workers 4
```

#### 3.2 在线评估(COCO test server)
```bash
cd test_online

python test_online.py \
    --model_path ../save_models/fsgr_full_best.pth \
    --vocab_path ../vocab.pkl \
    --text \
    --annotation_folder cocotest2014.json \
    --features_path ../../datasets/coco/images/test2014/ \
    --text_embed_path ../../MaskCLIP/pretrain/ram_ViT16_clip_text.pth \
    --pre_vs_path ../../.cache/clip/ViT-B-16.pt \
    --pre_name ViT-B/16 \
    --batch_size 10
```

---

## PMA-Net解码器融合方案

### 融合架构设计

FSGR使用标准Transformer解码器,而PMA-Net在解码器的自注意力层中集成了**原型记忆网络(Prototypical Memory)**。融合策略如下:

#### 1. 核心修改点

**PMA-Net的关键创新:**
- 在解码器的**每个自注意力层**中添加记忆机制
- 使用**memory banks**存储过去训练批次的key和value激活
- 通过**K-Means聚类**生成原型向量
- 在注意力计算中同时考虑当前输入和记忆原型

**FSGR的解码器特点:**
- 使用语义gap recovery机制
- 有cross-modal attention与CLIP视觉特征交互
- 已经有较强的语义对齐能力

#### 2. 融合实现步骤

**步骤1: 修改解码器自注意力层**

创建新文件 `models/fsgr/pma_decoder.py`:

```python
import torch
import torch.nn as nn
from models.fsgr.attention import MultiHeadAttention
from sklearn.cluster import MiniBatchKMeans
import faiss
import numpy as np

class PrototypicalMemoryAttention(nn.Module):
    """
    PMA-Net的原型记忆注意力层
    """
    def __init__(self, d_model=512, h=8, dropout=0.1, 
                 num_prototypes=1024, memory_bank_size=1500):
        super(PrototypicalMemoryAttention, self).__init__()
        
        self.d_model = d_model
        self.h = h
        self.num_prototypes = num_prototypes
        self.memory_bank_size = memory_bank_size
        
        # 标准多头注意力
        self.mhatt = MultiHeadAttention(d_model, h, dropout)
        
        # 记忆banks (存储过去的keys和values)
        self.register_buffer('memory_bank_k', torch.zeros(0, h, d_model // h))
        self.register_buffer('memory_bank_v', torch.zeros(0, h, d_model // h))
        
        # 原型向量
        self.register_buffer('prototype_keys', torch.zeros(num_prototypes, h, d_model // h))
        self.register_buffer('prototype_values', torch.zeros(num_prototypes, h, d_model // h))
        
        # Segment embeddings (区分输入和记忆)
        self.input_segment_emb = nn.Parameter(torch.randn(1, 1, d_model))
        self.memory_segment_emb = nn.Parameter(torch.randn(1, 1, d_model))
        
        # K-Means聚类器(用于原型生成)
        self.kmeans = None
        self.update_counter = 0
        self.update_stride = 10  # 每10步更新一次原型
        
    def update_memory_banks(self, new_keys, new_values):
        """
        更新记忆banks
        """
        # new_keys/values shape: (B, num_heads, seq_len, d_k)
        B, h, seq_len, d_k = new_keys.shape
        
        # 展平并添加到memory bank
        keys_flat = new_keys.reshape(-1, h, d_k)  # (B*seq_len, h, d_k)
        values_flat = new_values.reshape(-1, h, d_k)
        
        # 添加到bank
        self.memory_bank_k = torch.cat([self.memory_bank_k, keys_flat], dim=0)
        self.memory_bank_v = torch.cat([self.memory_bank_v, values_flat], dim=0)
        
        # 保持bank大小在限制内
        if self.memory_bank_k.size(0) > self.memory_bank_size:
            self.memory_bank_k = self.memory_bank_k[-self.memory_bank_size:]
            self.memory_bank_v = self.memory_bank_v[-self.memory_bank_size:]
    
    def generate_prototypes(self):
        """
        使用K-Means从memory banks生成原型
        """
        if self.memory_bank_k.size(0) < self.num_prototypes:
            return  # 数据不足,跳过
        
        # 对每个头分别聚类
        for head_idx in range(self.h):
            # 获取这个头的keys
            keys_head = self.memory_bank_k[:, head_idx, :].cpu().numpy()
            values_head = self.memory_bank_v[:, head_idx, :].cpu().numpy()
            
            # K-Means聚类
            kmeans = MiniBatchKMeans(
                n_clusters=self.num_prototypes,
                batch_size=min(10000, len(keys_head)),
                max_iter=10
            )
            labels = kmeans.fit_predict(keys_head)
            
            # 获取聚类中心作为原型keys
            prototype_k = torch.from_numpy(kmeans.cluster_centers_).to(keys_head.device)
            self.prototype_keys[:, head_idx, :] = prototype_k
            
            # 为每个原型计算对应的value(使用最近邻平均)
            for cluster_id in range(self.num_prototypes):
                mask = (labels == cluster_id)
                if mask.sum() > 0:
                    cluster_values = values_head[mask]
                    prototype_v = cluster_values.mean(axis=0)
                    self.prototype_values[cluster_id, head_idx, :] = \
                        torch.from_numpy(prototype_v).to(values_head.device)
    
    def forward(self, queries, keys, values, attention_mask=None):
        """
        前向传播
        queries, keys, values: (B, seq_len, d_model)
        """
        B, seq_len, d_model = queries.shape
        
        # 添加segment embeddings
        queries_with_seg = queries + self.input_segment_emb
        
        # 在训练时更新memory banks
        if self.training:
            # 获取当前batch的keys和values投影
            # 这里需要访问MultiHeadAttention内部的投影
            # 简化实现:直接存储输入
            with torch.no_grad():
                k_proj = keys.view(B, -1, self.h, self.d_model // self.h)
                v_proj = values.view(B, -1, self.h, self.d_model // self.h)
                k_proj = k_proj.transpose(1, 2)  # (B, h, seq_len, d_k)
                v_proj = v_proj.transpose(1, 2)
                
                self.update_memory_banks(k_proj, v_proj)
                
                # 定期更新原型
                self.update_counter += 1
                if self.update_counter % self.update_stride == 0:
                    self.generate_prototypes()
        
        # 将原型添加到keys和values
        if self.prototype_keys.size(0) > 0:
            # prototype_keys: (num_prototypes, h, d_k)
            # 需要reshape为 (B, num_prototypes, d_model)
            num_proto = self.prototype_keys.size(0)
            proto_k = self.prototype_keys.transpose(0, 1).reshape(self.h, num_proto, -1)
            proto_k = proto_k.transpose(0, 1).reshape(num_proto, -1)  # (num_proto, d_model)
            proto_k = proto_k.unsqueeze(0).expand(B, -1, -1)  # (B, num_proto, d_model)
            
            proto_v = self.prototype_values.transpose(0, 1).reshape(self.h, num_proto, -1)
            proto_v = proto_v.transpose(0, 1).reshape(num_proto, -1)
            proto_v = proto_v.unsqueeze(0).expand(B, -1, -1)
            
            # 添加memory segment embedding
            proto_k = proto_k + self.memory_segment_emb
            proto_v = proto_v + self.memory_segment_emb
            
            # 拼接
            extended_keys = torch.cat([keys, proto_k], dim=1)
            extended_values = torch.cat([values, proto_v], dim=1)
            
            # 扩展attention mask
            if attention_mask is not None:
                proto_mask = torch.zeros(B, 1, 1, num_proto, 
                                        device=attention_mask.device, 
                                        dtype=attention_mask.dtype)
                extended_mask = torch.cat([attention_mask, proto_mask], dim=-1)
            else:
                extended_mask = None
        else:
            extended_keys = keys
            extended_values = values
            extended_mask = attention_mask
        
        # 执行注意力
        output = self.mhatt(queries_with_seg, extended_keys, extended_values, 
                           extended_mask)
        
        return output


class PMATransformerDecoderLayer(nn.Module):
    """
    融合PMA的Transformer解码器层
    """
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, 
                 d_model=512, h=8, d_ff=2048, dropout=0.1,
                 num_prototypes=1024, memory_bank_size=1500,
                 use_pma_in_first_layer=True):
        super(PMATransformerDecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.N_dec = N_dec
        self.use_pma_in_first_layer = use_pma_in_first_layer
        
        # 词嵌入
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # 解码器层
        self.layers = nn.ModuleList()
        for i in range(N_dec):
            # 决定是否在这一层使用PMA
            use_pma = (i > 0) or use_pma_in_first_layer
            
            if use_pma:
                self_attn = PrototypicalMemoryAttention(
                    d_model, h, dropout, num_prototypes, memory_bank_size
                )
            else:
                self_attn = MultiHeadAttention(d_model, h, dropout)
            
            # Cross attention (与encoder输出交互)
            cross_attn = MultiHeadAttention(d_model, h, dropout)
            
            # Feed-forward
            ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
            
            self.layers.append(nn.ModuleDict({
                'self_attn': self_attn,
                'cross_attn': cross_attn,
                'ffn': ffn,
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            }))
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, encoder_output, caption_input, encoder_mask=None, caption_mask=None):
        """
        encoder_output: (B, seq_len_enc, d_model)
        caption_input: (B, seq_len_dec) - token ids
        """
        B, seq_len = caption_input.shape
        
        # 词嵌入 + 位置编码
        positions = torch.arange(seq_len, device=caption_input.device).unsqueeze(0)
        x = self.word_emb(caption_input) + self.pos_emb(positions)
        
        # 通过解码器层
        for layer in self.layers:
            # Self-attention (with PMA if applicable)
            residual = x
            x_norm = layer['norm1'](x)
            if isinstance(layer['self_attn'], PrototypicalMemoryAttention):
                x = layer['self_attn'](x_norm, x_norm, x_norm, caption_mask)
            else:
                x = layer['self_attn'](x_norm, x_norm, x_norm, caption_mask)
            x = residual + layer['dropout'](x)
            
            # Cross-attention
            residual = x
            x_norm = layer['norm2'](x)
            x = layer['cross_attn'](x_norm, encoder_output, encoder_output, encoder_mask)
            x = residual + layer['dropout'](x)
            
            # Feed-forward
            residual = x
            x_norm = layer['norm3'](x)
            x = layer['ffn'](x_norm)
            x = residual + layer['dropout'](x)
        
        # 输出投影
        logits = self.fc_out(x)
        
        return logits
```

**步骤2: 修改主模型文件**

修改 `models/fsgr/transformer.py`,添加PMA解码器选项:

```python
from models.fsgr.pma_decoder import PMATransformerDecoderLayer

class TransformerWithPMA(nn.Module):
    def __init__(self, bos_idx, encoder, decoder_type='standard', 
                 use_pma=False, pma_config=None, **kwargs):
        super(TransformerWithPMA, self).__init__()
        
        self.bos_idx = bos_idx
        self.encoder = encoder
        
        # 选择解码器类型
        if use_pma:
            self.decoder = PMATransformerDecoderLayer(
                vocab_size=kwargs.get('vocab_size'),
                max_len=kwargs.get('max_len', 54),
                N_dec=kwargs.get('N_dec', 3),
                padding_idx=kwargs.get('padding_idx'),
                d_model=kwargs.get('d_model', 512),
                h=kwargs.get('h', 8),
                num_prototypes=pma_config.get('num_prototypes', 1024),
                memory_bank_size=pma_config.get('memory_bank_size', 1500),
                use_pma_in_first_layer=pma_config.get('use_pma_in_first_layer', True)
            )
        else:
            self.decoder = decoder_type  # 使用原有的FSGR解码器
        
        # ... 其他初始化代码 ...
```

**步骤3: 创建训练脚本**

创建 `train_transformer_pma.py`:

```python
import argparse
import torch
from models.fsgr import TransformerEncoder, ScaledDotProductAttention
from models.fsgr.pma_decoder import PMATransformerDecoderLayer
from data import ImageDetectionsField, TextField, COCO
# ... 其他导入 ...

def train_pma_model():
    parser = argparse.ArgumentParser()
    
    # FSGR原有参数
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--annotation_folder', type=str, required=True)
    # ... 其他FSGR参数 ...
    
    # PMA特定参数
    parser.add_argument('--use_pma', action='store_true', 
                       help='Use PMA-Net decoder')
    parser.add_argument('--num_prototypes', type=int, default=1024,
                       help='Number of prototype vectors')
    parser.add_argument('--memory_bank_size', type=int, default=1500,
                       help='Size of memory banks')
    parser.add_argument('--pma_update_stride', type=int, default=10,
                       help='Update prototypes every N steps')
    parser.add_argument('--use_pma_first_layer', action='store_true',
                       help='Use PMA in first decoder layer')
    
    args = parser.parse_args()
    
    # 构建模型
    encoder = TransformerEncoder(2, 0, text=args.text,
                                attention_module=ScaledDotProductAttention)
    
    if args.use_pma:
        decoder = PMATransformerDecoderLayer(
            vocab_size=len(text_field.vocab),
            max_len=54,
            N_dec=3,
            padding_idx=text_field.vocab.stoi['<pad>'],
            num_prototypes=args.num_prototypes,
            memory_bank_size=args.memory_bank_size,
            use_pma_in_first_layer=args.use_pma_first_layer
        )
    else:
        from models.fsgr import TransformerDecoderLayer
        decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3,
                                         text_field.vocab.stoi['<pad>'])
    
    # ... 训练循环 ...

if __name__ == '__main__':
    train_pma_model()
```

#### 3. 训练融合模型

```bash
# 使用PMA解码器训练FSGR
python train_transformer_pma.py \
    --text \
    --use_pma \
    --exp_name fsgr_pma_fusion \
    --batch_size 64 \
    --workers 4 \
    --return_index \
    --features_path datasets/coco/images \
    --annotation_folder m2_annotations \
    --text_embed_path ../MaskCLIP/pretrain/ram_ViT16_clip_text.pth \
    --labels_path ../MaskCLIP/patch_labels.hdf5 \
    --pre_vs_path .cache/clip/ViT-B-16.pt \
    --pre_name ViT-B/16 \
    --num_prototypes 1024 \
    --memory_bank_size 1500 \
    --pma_update_stride 10 \
    --use_pma_first_layer
```

**注意事项:**
1. PMA需要更多GPU内存,建议减小batch_size(从100降到64或更低)
2. memory_bank_size=1500意味着存储约1.5M个样本的激活
3. 每10步更新一次原型(可调整以平衡性能和计算成本)

#### 4. 消融实验建议

为了验证融合效果,进行以下实验:

```bash
# 实验1: 基线FSGR(无PMA)
python train_transformer.py --text --exp_name fsgr_baseline ...

# 实验2: FSGR + PMA(所有层)
python train_transformer_pma.py --text --use_pma --use_pma_first_layer --exp_name fsgr_pma_all ...

# 实验3: FSGR + PMA(跳过第一层)
python train_transformer_pma.py --text --use_pma --exp_name fsgr_pma_nofirst ...

# 实验4: 不同原型数量
python train_transformer_pma.py --text --use_pma --num_prototypes 512 --exp_name fsgr_pma_512 ...
python train_transformer_pma.py --text --use_pma --num_prototypes 2048 --exp_name fsgr_pma_2048 ...

# 实验5: 不同记忆库大小
python train_transformer_pma.py --text --use_pma --memory_bank_size 1000 --exp_name fsgr_pma_mem1k ...
```

---

## 训练和评估

### 1. 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir=tensorboard_logs --port=6006

# 在浏览器中查看: http://localhost:6006
```

### 2. 评估指标

模型会在以下指标上评估:
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: n-gram精度
- **METEOR**: 基于词干和同义词的度量
- **ROUGE-L**: 最长公共子序列
- **CIDEr**: 共识度量(主要指标)
- **SPICE**: 基于场景图的语义度量

**预期性能(COCO Karpathy test split):**
- FSGR (ViT-L): CIDEr ~150.0
- FSGR + PMA: CIDEr ~152-154 (预期提升2-4点)

### 3. 模型集成(可选)

训练多个不同种子的模型进行集成:

```bash
# 训练3个模型
for seed in 99 42 3407; do
    python train_transformer_pma.py \
        --text --use_pma \
        --exp_name fsgr_pma_seed${seed} \
        --seed ${seed} \
        --batch_size 64 \
        ... 其他参数 ...
done

# 集成评估
python test_ensemble.py \
    --model_paths save_models/fsgr_pma_seed99.pth \
                  save_models/fsgr_pma_seed42.pth \
                  save_models/fsgr_pma_seed3407.pth \
    --vocab_path vocab.pkl \
    --text \
    --use_pma \
    ... 其他参数 ...
```

---

## 故障排除

### 常见问题

1. **CUDA Out of Memory**
   ```bash
   # 减小batch size
   --batch_size 32  # 或更小
   
   # 使用梯度累积
   --gradient_accumulation_steps 4
   
   # 减小原型数量
   --num_prototypes 512
   ```

2. **MaskCLIP文本嵌入维度不匹配**
   - 确保提取的文本嵌入维度与CLIP模型匹配(ViT-B/16: 512维)
   - 检查 `text_embed_path` 指向的文件格式

3. **M2标注文件格式错误**
   - 确保使用正确的JSON格式
   - 参考RSTNet仓库的标注格式

4. **训练不收敛**
   - 检查学习率(默认2.5e-4可能需要调整)
   - 确保数据预处理正确
   - 尝试预训练语言模型

### 性能优化

1. **使用混合精度训练**
   ```python
   # 在训练脚本中添加
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       output = model(...)
       loss = criterion(output, target)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

2. **使用更大的GPU**
   - V100 (32GB): batch_size 100+
   - A100 (40GB/80GB): batch_size 150+
   - RTX 3090 (24GB): batch_size 64-80

---

## 理论创新点总结

### FSGR的贡献
1. **语义gap恢复**: 使用CLIP对齐视觉和语义特征
2. **细粒度语义对齐**: patch-level的对比学习
3. **参数高效tuning**: 只微调部分层

### PMA-Net的贡献
1. **原型记忆网络**: 存储和检索过去训练样本的激活
2. **动态原型生成**: 通过K-Means聚类生成代表性原型
3. **减少幻觉**: 通过检索相似样本改善描述质量

### 融合的优势
1. **互补性强**: FSGR的语义对齐 + PMA的记忆检索
2. **性能提升**: 预期在CIDEr上提升2-4点
3. **创新性**: 首次将原型记忆应用于语义gap recovery

---

## 下一步工作

1. **完成FSGR基线复现**: 确保能达到论文报告的性能
2. **实现PMA解码器**: 根据上述代码框架实现
3. **消融实验**: 验证每个组件的贡献
4. **性能优化**: 调整超参数以达到最佳性能
5. **论文撰写**: 记录实验结果和创新点

---

## 参考资源

### 官方仓库
- **FSGR**: https://github.com/gjc0824/FSGR
- **PMA-Net**: https://github.com/aimagelab/PMA-Net
- **MaskCLIP**: https://github.com/chongzhou96/MaskCLIP
- **RSTNet**: https://github.com/zhangxuying1004/RSTNet

### 数据集
- **COCO**: https://cocodataset.org/#download
- **nocaps**: https://nocaps.org/download

### 预训练模型
- **CLIP**: https://github.com/openai/CLIP
- **ViT-B/16**: https://openaipublic.azureedge.net/clip/models/

### 论文
- FSGR: "Fully Semantic Gap Recovery for End-to-End Image Captioning"
- PMA-Net: "With a Little Help from your own Past: Prototypical Memory Networks for Image Captioning"
- CLIP: "Learning Transferable Visual Models From Natural Language Supervision"

---

祝复现顺利!如有任何问题,欢迎随时询问。
