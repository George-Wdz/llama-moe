<div align="center">
  <h1>LLaMA-MoE：基于 LLaMA 的持续预训练 Mixture-of-Experts 构建方法</h1>
  <img src="docs/imgs/title-favicon.png" width="200" alt="LLaMA-MoE favicon" style="border-radius: 5%;"><br />
  <span style="color:red">📢 <strong><i>面向所有人的更小、更经济的 MoE 模型！</i></strong></span>
  <div>
    <a href="https://huggingface.co/llama-moe" target="_blank">🤗 模型权重</a> | <a href="#quick-start">🚀 快速开始</a> | <a href="#installation">⚙️ 安装指南</a> | <a href="#expert-construction">🚧 专家构建</a> | <a href="#continual-pretraining">🚅 持续预训练</a> | <a href="#evaluation">💎 评测</a> | <a href="#sft">💬 监督微调（SFT）</a>
  </div>
  <a href="docs/LLaMA_MoE.pdf" target="_blank"><strong>📃 技术报告</strong></a>
</div>

<h2 id="llama-moe">🎉 简介</h2>

LLaMA-MoE 是一系列开源的 Mixture-of-Experts（MoE）模型，基于 [LLaMA](https://github.com/facebookresearch/llama) 和 [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) 构建。
我们通过以下两个步骤完成 LLaMA-MoE 的构建：
1. 将 LLaMA 的 FFN 划分为稀疏专家，并为每层专家插入 top-K gate。
2. 使用来自 [Sheared LLaMA](https://arxiv.org/abs/2310.06694) 的优化数据采样权重，以及来自 [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) 的过滤数据集，对初始化后的 MoE 模型进行持续预训练。

![MoE Routing](./docs/imgs/MoE-Routing.gif)

<h2 id="features">🔥 特性</h2>

1. **轻量级模型**：激活参数量仅为 3.0~3.5B，适合部署和研究使用。
2. **多种专家构建方式**：
   1. Neuron-Independent：Random、Clustering、Co-activation Graph、Gradient ([Zhang et al., 2022](http://arxiv.org/abs/2110.01786), [Zuo et al., 2022](http://arxiv.org/abs/2204.07675))
   2. Neuron-Sharing：Inner、Inter（residual）
3. **多种 MoE 门控策略**：
   1. TopK Noisy Gate ([Shazeer et al., 2017](http://arxiv.org/abs/1701.06538))
   2. Switch Gating ([Fedus et al., 2022](http://arxiv.org/abs/2101.03961))
4. **高效持续预训练**：
   1. 集成 FlashAttention-v2 ([Dao, 2023](https://github.com/Dao-AILab/flash-attention))
   2. 快速流式数据加载
5. **丰富的监控项**：
   1. Gate load、gate importance
   2. 按 step 的 loss、按 token 的 loss、balance loss
   3. TGS（tokens/GPU/second）、MFU（model FLOPs utilization）
   4. 其他可视化工具
6. **动态权重采样**：
   1. 自定义静态采样权重
   2. Sheared LLaMA 的动态 batch 加载 ([Xia et al., 2023](http://arxiv.org/abs/2310.06694))


<h2 id="quick-start">🚀 快速开始</h2>

```python
# python>=3.10

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.to("cuda:0")

input_text = "Suzhou is famous of"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = inputs.to("cuda:0")

pred = model.generate(**inputs, max_length=50, temperature=0.0)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
# Suzhou is famous of its beautiful gardens. The most famous one is the Humble Administrator's Garden. It is a classical Chinese garden with a history of more than 600 years. The garden is divided into three
```

<h2 id="installation">⚙️ 安装</h2>

1. 准备 conda 环境：`conda create -n smoe python=3.11`（如果你的环境名不是 `smoe`，则在启动脚本中需要同步修改环境名）
2. 在 `~/.bashrc` 中添加正确的环境变量（安装 `flash-attn` 时会使用较新的 `gcc` 版本）。例如：
    ```bash
    export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH
    export PATH=/mnt/petrelfs/share/gcc-10.1.0/bin:$PATH
    export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc-10.1.0/lib64:$LD_LIBRARY_PATH
    ```
3. 让环境变量生效：`source ~/.bashrc`
4. 安装 PyTorch（CUDA-11.8）：`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
5. 安装依赖：`pip install -r requirements.txt`
6. 安装 `flash-attn`：`pip install flash-attn==2.0.1 --no-build-isolation`。如遇到构建问题，可能需要参考 [flash-attn 安装说明](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features)。
7. 安装最新版 Git：`conda install git`
8. 克隆仓库：`git clone git@github.com:pjlab-sys4nlp/llama-moe.git`（如果你没有配置 GitHub SSH key，可能无法通过 SSH 克隆。可参考 [文档](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)）。
9. 进入目录：`cd llama-moe`
10. 以 [editable mode](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-e) 安装 `smoe`：`pip install -e .[dev]`
11. 安装 `pre-commit` 钩子：`pre-commit install`

<h2 id="performance">📊 模型性能</h2>

| Model                     | \#Activated Experts | \#Experts | \#Activated Params |                         Foundation Model                          |                              SFT Model                               |
| :------------------------ | :-----------------: | :-------: | :----------------: | :---------------------------------------------------------------: | :------------------------------------------------------------------: |
| **LLaMA-MoE-3.0B**        |          2          |    16     |        3.0B        | [🤗 base](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_0B-2_16) | [🤗 SFT](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_0B-2_16-sft) |
| **LLaMA-MoE-3.5B (4/16)** |          4          |    16     |        3.5B        | [🤗 base](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-4_16) | [🤗 SFT](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-4_16-sft) |
| **LLaMA-MoE-3.5B (2/8)**  |          2          |     8     |        3.5B        | [🤗 base](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-2_8)  | [🤗 SFT](https://huggingface.co/llama-moe/LLaMA-MoE-v1-3_5B-2_8-sft)  |

- 基础模型

| Model                                                                                 | Average  |   SciQ   |   PIQA   | WinoGrande |  ARC-e   | ARC-c (25) | HellaSwag (10) |  LogiQA  | BoolQ (32) | LAMBADA  | NQ (32)  | MMLU (5) |
| :------------------------------------------------------------------------------------ | :------: | :------: | :------: | :--------: | :------: | :--------: | :------------: | :------: | :--------: | :------: | :------: | :------: |
| [OPT-2.7B](https://huggingface.co/facebook/opt-2.7b)                                  |   50.3   |   78.9   |   74.8   |    60.8    |   54.4   |    34.0    |      61.4      |   25.8   |    63.3    |   63.6   |   10.7   |   25.8   |
| [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b)                          |   51.5   |   83.2   |   73.6   |    59.6    |   58.8   |    36.7    |      60.7      |   28.1   |    65.9    |   64.6   |   8.7    |   26.8   |
| [INCITE-BASE-3B](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1) |   53.7   |   85.6   |   73.9   |    63.5    |   61.7   |    40.3    |      64.7      |   27.5   |    65.8    |   65.4   |   15.2   |   27.2   |
| [Open-LLaMA-3B-v2](https://huggingface.co/openlm-research/open_llama_3b_v2)           |   55.6   |   88.0   |   77.9   |    63.1    |   63.3   |    40.1    |      71.4      |   28.1   |    69.2    |   67.4   |   16.0   |   26.8   |
| [Sheared-LLaMA-2.7B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-2.7B)         |   56.4   |   87.5   |   76.9   |    65.0    |   63.3   |    41.6    |      71.0      |   28.3   |    73.6    |   68.3   |   17.6   | **27.3** |
| **LLaMA-MoE-3.0B**                                                                    |   55.5   |   84.2   |   77.5   |    63.6    |   60.2   |    40.9    |      70.8      | **30.6** |    71.9    |   66.6   |   17.0   |   26.8   |
| **LLaMA-MoE-3.5B (4/16)**                                                             | **57.7** |   87.6   | **77.9** |    65.5    | **65.6** |  **44.2**  |    **73.3**    |   29.7   |  **75.0**  | **69.5** | **20.3** |   26.8   |
| **LLaMA-MoE-3.5B (2/8)**                                                              |   57.6   | **88.4** |   77.6   |  **66.7**  |   65.3   |    43.1    |    **73.3**    |   29.6   |    73.9    |   69.4   |   19.8   |   27.0   |

- SFT 模型

| Model                                  | MMLU  | ARC-c | HellaSeag | TruthfulQA | MT-Bench |
| :------------------------------------- | :---: | :---: | :-------: | :--------: | :------: |
| Sheared LLaMA-2.7B ShareGPT            | 28.41 | 41.04 |   71.21   |   47.65    |   3.79   |
| Sheared LLaMA-2.7B Deita6K (Our Impl.) | 25.24 | 43.69 |   71.70   |   49.00    |   4.06   |
| LLaMA-MoE-v1-3.0B (2/16)               | 23.61 | 43.43 |   72.28   |   44.24    |   4.15   |
| LLaMA-MoE-v1-3.5B (4/16)               | 26.49 | 48.29 |   75.10   |   45.91    |   4.60   |
| LLaMA-MoE-v1-3.5B (2/8)                | 25.53 | 45.99 |   74.95   |   44.39    |   4.72   |

<h2 id="expert-construction">🚧 专家构建</h2>

- Neuron-Independent
  - Independent<sub>Random</sub>：`bash ./scripts/expert_construction/split/run_split_random.sh`
  - Independent<sub>Clustering</sub>：`bash ./scripts/expert_construction/split/run_split_clustering.sh`
- Neuron-Sharing
  - Sharing<sub>Inner</sub>：`bash ./scripts/expert_construction/split/run_split_gradient.sh`
  - Sharing<sub>Inter</sub>：`bash ./scripts/expert_construction/split/run_split_gradient_residual.sh`

更多信息请参考 [专家构建文档](docs/expert_construction/README.md)。

<h2 id="continual-pretraining">🚅 持续预训练</h2>


### 分词

将 [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) 下载到 `/path_to_data`，并把不同领域的数据放入不同文件夹中：
  - `/path_to_data/en_arxiv`
  - `/path_to_data/en_book`
  - `/path_to_data/en_c4`
  - `/path_to_data/en_cc`
  - `/path_to_data/en_stack`
  - `/path_to_data/en_wikipedia`
  - `/path_to_data/github`

每个文件都应以 `*.jsonl` 结尾，并且每一行形如：
```
{"id": "id-info", "content": "raw text to be tokenized"}
```

在每个文件夹中运行下面的命令进行分词：

```bash
python -m smoe.utils.tokenize \
  -f jsonl \
  -t /path_to_tokenizer \
  -i /path_to_data/en_arxiv \
  -o /path_to_data_tokenized/en_arxiv
```

### 持续预训练（CPT）

- **注意：** 请手动创建 `logs/` 文件夹：`mkdir -p logs`
- 有关持续预训练的运行方式，请查看 [CPT 文档](docs/continual_pretraining/README.md)。

<h2 id="evaluation">💎 评测</h2>

- 关于 Natural Questions（NQ）的评测，请参考 [opencompass](https://github.com/Spico197/opencompass/tree/main)。
- 其他任务的评测，请参考 [lm-eval-harness](https://github.com/spico197/smoe-eval)。

<h2 id="sft">💬 监督微调（SFT）</h2>

我们提供了用于构建聊天机器人的简单 SFT 示例。
更多细节请参考 [SFT 文档](docs/supervised_fine_tuning/SFT.md) 和 `scripts/sft`。

<h2 id="citation">📑 引用</h2>

```bibtex
@article{llama-moe,
  title={LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training},
  author={Tong Zhu and Xiaoye Qu and Daize Dong and Jiacheng Ruan and Jingqi Tong and Conghui He and Yu Cheng},
  journal={arXiv preprint arXiv:2406.16554},
  year={2024},
  url={https://arxiv.org/abs/2406.16554},
}
```

<hr>
<p align="center">LLaMA-MoE Team w/ ❤️</p>