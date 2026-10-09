# AL4QED：可复现的主动学习实验代码

主线是学习指定数值后端产生的、截断至 [0,1] 的可见度目标，并比较相同标签预算下的随机采样与主动学习。

## 首先运行

建议在独立 Python 环境中：

```bash
python -m pip install -e ".[ml,dev]"
python -m pytest -q tests
```

无需 SDP 的流程验证（2×2，PPT 给出精确的截断可分可见度）：

```bash
python -m al4qed.train --mode compare --model nn --backend ppt --dims 2 2 \
  --initial 30 --pool 300 --cycles 5 --queries 15 --epochs 100 \
  --test 200 --val 40 --seeds 0 1 2 \
  --strategies random uncertainty boundary --output results/ppt_mlp
```

随机森林对照：把 `--model nn` 改成 `--model rf`，使用新的输出目录。SVR、Ridge 和可选 XGBoost 支持 random/margin；它们不假装具有 MC Dropout 不确定度。首次运行默认 CPU。Mac 也可以显式使用 `--device mps`，但当前验证环境只测试 CPU。

已有 3×3 本地复现算法仍可调用：

```bash
python -m al4qed.train --mode compare --model nn --backend local --dims 3 3 \
  --initial 30 --pool 500 --cycles 5 --queries 15 --epochs 100 \
  --test 200 --val 40 --seeds 0 1 2 --oracle-N 100 --oracle-iters 10 \
  --output results/local_3x3
```

这只是你原先算法的修正版，**不是作者官方实现**。求解器顺序为可用的 MOSEK、CLARABEL、SCS；求解失败或未通过数值检查时尝试下一个可用求解器。无效或不精确求解会报错，不能静默变成 chi=0。3×3 标签成本较高，先用小预算做连接测试。

## 直接接入 Ohst 作者源码

论文参考文献 [35] 的地址：https://gitlab.com/tqo/quantum-correlations 。已检查的版本：

- 主仓库：`035d401fc239339ae95104170ce15c60dbe26567`
- `lib/kvant`：`44caf9384075b5db3f3b8efe1e469947f4a08287`
- 真正的数值实现是 Julia / Convex.jl，不是 Python。
- 当前 README 的 `EntanglementRobustness` 示例与实际顶层函数签名不一致，且根目录另有未配置 URL 的 `kvant` gitlink。因此桥接直接调用实际存在的 `RandomBlochPolytope` 和 `RobustnessToSeparabilityByBlochPolytope`，仅初始化需要的 `lib/kvant`。

安装依赖并获取固定版本：

```bash
python -m pip install -r requirements-ohst.txt
python -m al4qed.setup_ohst ../quantum-correlations --install-julia-deps
export OHST_REPO="$(cd ../quantum-correlations && pwd)"
export OHST_SOLVER=SCS
```

作者接口已通过实际 Bell 态调用和 3×3 Horodecki 主动学习闭环验证，详见 docs/VALIDATION.md。首次启动 juliacall 可能下载 Julia 并编译依赖。即使使用 SCS，上游文件仍无条件导入 Mosek/MosekTools，因此包也需安装；选择 MOSEK 求解则需有效许可证。

运行：

```bash
python -m al4qed.train --mode compare --model nn --backend external \
  --external al4qed.ohst_bridge:query \
  --backend-version ohst-035d401fc239339ae95104170ce15c60dbe26567-bridge-v1-SCS \
  --dims 3 3 --initial 30 --pool 500 --cycles 5 --queries 15 \
  --test 200 --val 40 --seeds 0 1 2 --oracle-N 100 --oracle-iters 10 \
  --output results/ohst_3x3
```

若使用 MOSEK，同时更改 `OHST_SOLVER` 和版本参数末尾的 `SCS`。不接受不同源码版本或对相关源码的未记录修改。桥接在同一个 Python 进程里复用 Julia。作者函数会替换低权重因子且不返回完整求解状态，所以桥接在作者最终适配的多面体上额外求解一次、检查状态、重构残差和 PSD；这一步是数值验证，不是重新实现自适应算法。此额外开销计入后端耗时。上游没有可靠的收敛日志接口，因此不虚构收敛历史；桥接的 `converged=False` 表示没有报告已验证的收敛信息。

作者默认优化 t≥0，而本项目学习 min(1,t)。对 I/d，上游未截断问题无界，桥接显式返回截断值 1。最终输出是经过数值检查的内逼近下界估计，不是精确算术证明。

## 此次修正的科学含义

`chi_lower < 1` 不能推出纠缠。内多面体没有找到当前态的分解，只能说明此次内逼近不够大。

- 连续目标：固定后端、顶点数、迭代预算和随机初始化下的可见度估计。
- `ENTANGLED`：当前双体接口只根据显著 NPT 判定。
- `SEPARABLE`：当前接口只在 2×2 / 2×3 的 PPT 充分条件下输出；对高维内多面体数值结果保守输出 UNKNOWN。
- `UNKNOWN`：未获得分类证据，不作为纠缠或可分标签训练。
- 模型输出没有认证资格。MC Dropout / 树间分歧也不是经过校准的置信区间。
- boundary 的 0.99 是数值目标的采样水平，不是被证明的 SEP 边界；默认比较中也保留 random 和 uncertainty。
- 标签误差与模型误差是两件事。对有限多面体结果的低 MAE 不代表对真实可见度也有低 MAE。

之前上传的独立 `test.py` 存在 BSEP 内逼近不可行及求解异常被误标为 GME 的问题；新训练主线不使用它。双体接口不能直接认证三体 GME。`ohst_reproduction.py` 保留原文件，不参与新训练主线，不是作者仓库代码，也未在本轮验证。

Zhang 的局域幺正增强针对真实分类标签有依据。但有限预算多面体估计可能随初始化/坐标而变化，不能把其回归标签无条件复制给 LU 增强态。本轮没有添加这样的错误增强，也没有把伪标签当作 SDP 认证结果。

## 实验协议与输出

- 固定独立测试集；每个 seed 下所有策略共用初始集、验证集、候选池与初始化。
- 仅查询选中的候选态；测试集与验证集单独标注，预算单独报告。
- 每轮从同一初始模型重新训练；标准化仅拟合当前训练集，最佳权重通过固定验证集选择。
- 每次查询后重新训练，包括最后一批；`cycle=0` 是初始模型。
- 使用状态内容及完整后端配置生成确定性 oracle seed；缓存区分维数、数值设置、后端版本。缓存命中仍计入逻辑标签预算，不算新的后端调用。
- 不用测试集决定提前停止或采样；多种策略结果只是对照结果，不能在同一测试集反复调参后称为最终泛化性能。
- `mixed` 是明确的合成分布：Ginibre 态、可分乘积态凸组合、随机纯态加白噪声各约 1/3。它不代表所有量子态上的均匀分布，也未保证各类别平衡。

文件：`config.json`（参数、依赖版本、源码哈希），`test_set.npz`、每个 seed 的原始态划分，`learning_curves.csv`（误差与预算），`summary.json`（均值与样本标准差），`accounting.json`（后端调用/缓存），学习曲线，以及每轮覆盖更新的模型、标准化器、标签集、查询日志。模型按每轮保存，但尚未提供中断后自动续训。

同一路径已有 `config.json` 时会拒绝覆盖。失败后换新的 `--output`，可以用 `--cache 旧目录/oracle_cache` 复用已成功验证的标签。缓存配置变化会自动失配，不会把其他算法设置的标签混入当前实验。

## 验证结果

见 `docs/VALIDATION.md`。`validation/` 中的小实验摘要用于检验实现与比较流程，不能作为论文的样本效率结论。正式研究应扩展状态分布、随机种子与标签预算，并对一部分测试态用更高精度/预算校验 oracle 偏差。

接口兼容说明：保留原来常用的 `--mode active --model nn --strategy boundary --initial ... --cycles ... --queries ... --epochs ...` 命令形式。`train.py` 和 `benchmarks.py` 的旧 Python 函数式入口已统一为 `run_experiment(args)`；`--mode supervised` 现在表示同一预算序列下的随机监督基线，不再采用旧的固定 `--samples` 接口。

## 仓库结构

- `src/al4qed/`：算法适配、特征、模型和训练入口。
- `tests/`：25 项回归测试，运行 `python -m pytest -q`。
- `docs/VALIDATION.md`：已执行测试及其范围。
- `validation/`：小实验的配置、CSV 与结果摘要；不提交模型权重和缓存。
- `figures/`、`papers/`、`patches/`：既有研究资料。历史图表不是新流程的验证结果。

常用命令：`python -m al4qed.train --help`、`python -m al4qed.benchmarks --help`。

## Two-qutrit G/W mixture

A new `--distribution ghz_w_3x3 --dims 3 3` experiment uses an explicitly defined G/W/white-noise mixture on two qutrits. It is not a three-party GHZ/W classification task. See [definition and commands](docs/GHZW_3X3.md) and [run results](docs/GHZW_3X3_RESULTS.md). The paired pilot entry point is `python -m al4qed.ghzw_compare --backend external --output results/ghzw_official`.
