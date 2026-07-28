# ============================================================================
# 训练脚本 - 用于训练机器人策略模型
#
# 架构说明：
# - 外部库：JAX/Flax (深度学习框架), Optax (优化器), wandb (实验追踪)
# - 内部模块：openpi.* (项目自定义的模型、数据加载器、工具函数等)
#
# 用途：基于模仿学习训练机器人策略，从专家演示中学习动作策略
# ============================================================================

# ============================================================================
# 第一部分：外部库导入
# ============================================================================

# Python 标准库
import dataclasses      # [标准库] 数据类装饰器和工具
import functools        # [标准库] 高阶函数工具（如 partial）
import logging          # [标准库] 日志记录
import platform         # [标准库] 获取系统平台信息
from typing import Any  # [标准库] 类型注解

# JAX 生态系统 (Google 开源的 ML 框架，类似 PyTorch/TensorFlow)
# 官网：https://github.com/google/jax
import etils.epath as epath                # [外部] 增强的路径处理库
import flax.nnx as nnx                     # [外部] Flax Neural Network eXplorer - JAX 的神经网络 API
from flax.training import common_utils     # [外部] Flax 训练工具
import flax.traverse_util as traverse_util # [外部] Flax 树结构遍历工具
import jax                                 # [外部] JAX 核心库（可微编程）
import jax.experimental                    # [外部] JAX 实验性功能
import jax.numpy as jnp                    # [外部] JAX 版本的 NumPy（支持 GPU/TPU 和自动微分）
import numpy as np                         # [外部] 传统 NumPy（CPU 数组计算）
import optax                               # [外部] Optax - JAX 的优化器库（SGD, Adam 等）

# 进度条和实验追踪工具
import tqdm_loggable.auto as tqdm          # [外部] tqdm 进度条（支持日志记录）
import wandb                               # [外部] Weights & Biases - ML 实验追踪平台

# ============================================================================
# 第二部分：项目内部模块导入
# ============================================================================

import openpi.models.model as _model                    # [内部] 模型定义（BaseModel, Observation, Actions）
import openpi.shared.array_typing as at                 # [内部] 类型注解和类型检查工具
import openpi.shared.nnx_utils as nnx_utils             # [内部] Flax NNX 工具函数
import openpi.training.checkpoints as _checkpoints      # [内部] 检查点保存/恢复逻辑
import openpi.training.config as _config                # [内部] 训练配置类 (TrainConfig)
import openpi.training.data_loader as _data_loader      # [内部] 数据加载器
import openpi.training.optimizer as _optimizer          # [内部] 优化器创建逻辑
import openpi.training.sharding as sharding             # [内部] FSDP 分片策略（多GPU分布式训练）
import openpi.training.utils as training_utils         # [内部] 训练工具（TrainState 定义等）
import openpi.training.weight_loaders as _weight_loaders # [内部] 预训练权重加载器


# ============================================================================
# 第三部分：工具函数
# ============================================================================

def init_logging():
    """自定义日志格式，提高可读性。

    功能：
    - 将日志级别缩写为单字母（D/I/W/E/C）
    - 格式化输出包含时间戳、消息、进程ID和代码位置
    """
    # 日志级别到单字母的映射表
    # D=DEBUG, I=INFO, W=WARNING, E=ERROR, C=CRITICAL
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    # 自定义日志格式化器（继承自 logging.Formatter）
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # 将完整的日志级别名称替换为单字母缩写
            # 例如: "DEBUG" -> "D", "INFO" -> "I"
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    # 创建自定义格式化器实例
    # 格式说明:
    # %(asctime)s.%(msecs)03d - 时间戳.毫秒(3位数字)
    # [%(levelname)s] - 日志级别(单字母)
    # %(message)-80s - 日志消息(左对齐，占80字符宽)
    # %(process)d - 进程ID
    # %(filename)s:%(lineno)d - 文件名:行号
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",  # 时间格式: 时:分:秒
    )

    # 获取根日志记录器 [标准库 logging]
    logger = logging.getLogger()
    # 设置日志级别为INFO（INFO及以上级别会被输出）
    logger.setLevel(logging.INFO)
    # 将自定义格式化器应用到根logger的第一个处理器
    # handlers[0] 通常是标准输出流处理器
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """初始化 Weights & Biases (wandb) 实验追踪工具。

    wandb [外部库] 是一个第三方实验追踪平台，用于记录训练指标、超参数和输出。
    官网: https://wandb.ai/
    安装: pip install wandb

    Args:
        config: 训练配置对象 [内部] openpi.training.config.TrainConfig
        resuming: 是否恢复之前的训练运行
        log_code: 是否记录代码到wandb（便于复现）
        enabled: 是否启用wandb（False则禁用）

    Raises:
        FileNotFoundError: 检查点目录不存在时抛出
    """
    # 如果wandb被禁用，则以disabled模式初始化（不发送数据到服务器）
    if not enabled:
        wandb.init(mode="disabled")
        return

    # 获取检查点目录路径
    ckpt_dir = config.checkpoint_dir
    # 验证检查点目录存在，否则抛出异常
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    # 恢复训练模式：从之前中断的地方继续
    if resuming:
        # 从检查点目录读取之前保存的wandb运行ID
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        # 使用相同的run_id恢复wandb运行，确保指标继续记录到同一个实验中
        # resume="must" [wandb参数] 表示必须存在对应的运行才能恢复
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 新训练运行：创建新的wandb实验
        wandb.init(
            name=config.exp_name,                           # 实验名称（如：act_real_imag_100k）
            config=dataclasses.asdict(config),              # 将配置对象转为字典记录 [标准库]
            project=config.project_name,                    # 项目名称（如：openpi）
        )
        # 保存wandb运行ID到检查点目录，以便后续恢复训练时使用
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    # 可选：记录代码到wandb，便于在wandb界面上查看完整的代码快照
    if log_code:
        # 记录项目根目录的代码（__file__的父目录的父目录）
        # epath [外部] 是增强的路径库
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """加载并验证预训练权重。

    Args:
        loader: 权重加载器 [内部] openpi.training.weight_loaders.WeightLoader
                支持从预训练模型（如 CLIP, ViT）加载部分权重
        params_shape: 模型参数的形状结构 [内部] openpi.shared.array_typing.Params

    Returns:
        加载的权重参数字典

    Raises:
        ValueError: 如果加载的权重形状或数据类型不匹配
    """
    # 从权重加载器加载权重（可能是部分权重）
    # loader.load() [内部] 根据配置从预训练模型加载参数
    loaded_params = loader.load(params_shape)

    # 验证加载的权重与模型期望的权重形状和类型一致
    # check_pytree_equality [内部] 自定义的 pytree 验证函数
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # 从加载的参数中移除 jax.ShapeDtypeStruct（占位符）
    # traverse_util [外部 flax] 用于扁平化和还原嵌套字典
    # jax.ShapeDtypeStruct [外部 JAX] 是形状/类型描述符，不是实际数据
    # 这确保只返回实际加载的参数
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


# ============================================================================
# 第四部分：训练状态初始化
# ============================================================================

@at.typecheck  # [内部] 装饰器：用于在运行时进行类型检查，确保传入的参数符合预期。
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """初始化训练状态（模型参数、优化器状态等）。

    TrainState [内部] 定义在 openpi.training.utils，包含：
    - step: 当前训练步数
    - params: 模型参数
    - model_def: 模型结构定义（Flax graphdef）
    - tx: 优化器（Optax GradientTransformation）
    - opt_state: 优化器状态（如 Adam 的动量）
    - ema_decay: EMA 衰减系数
    - ema_params: EMA 参数（用于推理）

    Args:
        config: 训练配置对象，包含优化器、学习率、模型结构等信息。[内部]
        init_rng: JAX 的随机数生成键，用于模型权重的随机初始化 [JAX PRNGKey]
        mesh:[外部 JAX] JAX 的设备网格，用于分布式并行计算（如 FSDP）
        resume: 是否恢复训练（如果是，则跳过初始化）

    Returns:
        (train_state, state_sharding): 训练状态和其分片策略
    """
    # 创建优化器 [内部] 封装 Optax 优化器和学习率调度
    # config.optimizer: 优化器类型（如 "adam"）
    # config.lr_schedule: 学习率调度（如 "cosine"）
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        """内部函数：初始化模型和训练状态。

        Args:
            rng: 随机数键
            partial_params: 可选的预训练权重（部分参数）
        """
        # 分裂随机数键 [外部 JAX] 用于模型初始化
        rng, model_rng = jax.random.split(rng)

        # 初始化模型（根据配置创建模型实例）
        # config.model.create [内部] 根据模型类型（如 ACT）创建模型
        model = config.model.create(model_rng)

        # 如果有预训练权重，将其合并到模型中
        if partial_params is not None:
            # nnx.split [外部 Flax] 将模型拆分为结构图和状态
            graphdef, state = nnx.split(model)
            # 将预训练权重（纯字典格式）合并到状态中
            # replace_by_pure_dict [外部 Flax] 用字典值替换状态中的对应参数
            state.replace_by_pure_dict(partial_params)
            # nnx.merge [外部 Flax] 重新组合模型
            model = nnx.merge(graphdef, state)

        # 提取模型参数 [外部 Flax]
        params = nnx.state(model)

        # 将冻结的参数转换为 bfloat16 精度
        # config.freeze_filter [内部] 过滤器，指定哪些参数需要冻结
        # nnx_utils.state_map [内部] 对匹配的参数应用转换函数
        # bfloat16 [外部 JAX] 节省显存的半精度浮点格式
        # 通常是为了在保持推理精度的同时减少显存占用。
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        # 构造并返回训练状态 [内部 dataclass]
        return training_utils.TrainState(
            step=0,                                          # 初始步数
            params=params,                                    # 模型参数（包含冻结和可训练的）
            model_def=nnx.graphdef(model),                   # 模型结构定义
            tx=tx,                                           # 优化器
            opt_state=tx.init(params.filter(config.trainable_filter)),  # 优化器状态（只初始化可训练参数）
            ema_decay=config.ema_decay,                      # EMA 衰减系数（如 0.999）
            ema_params=None if config.ema_decay is None else params,  # EMA 参数
        )

    # jax.eval_shape [外部 JAX] 计算函数输出的形状结构，不执行实际计算
    # 这用于获取 train_state 的形状结构，而不分配实际内存
    train_state_shape = jax.eval_shape(init, init_rng)

    # 创建 FSDP (Fully Sharded Data Parallel) 分片策略
    # sharding.fsdp_sharding [内部] 根据形状和 mesh 创建分片配置
    # 这将大模型参数分布到多个 GPU 上
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    # 如果是恢复训练，直接返回形状和分片（稍后从检查点加载实际值）
    if resume:
        return train_state_shape, state_sharding

    # 加载预训练权重（如果有）
    # config.weight_loader [内部] 权重加载器配置
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())

    # 创建复制分片（所有设备上的参数相同）
    # jax.sharding [外部 JAX] 分片相关 API
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 使用 JIT 编译初始化函数并执行
    # jax.jit [外部 JAX] 即时编译，加速执行
    # donate_argnums=(1,) [外部 JAX] 捐赠 partial_params 缓冲区（重用内存）
    # in_shardings/out_shardings [外部 JAX] 指定输入输出的分片策略
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # 捐赠 partial_params 的内存
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# ============================================================================
# 第五部分：训练步骤
# ============================================================================

@at.typecheck  # [内部] 运行时类型检查装饰器
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """执行单步训练：前向传播、反向传播、参数更新。

    Args:
        config: 训练配置 [内部]
        rng: 随机数键 [JAX PRNGKey]
        state: 当前训练状态 [内部 TrainState]
        batch: 训练批次 (observation, actions) [内部类型]
            - observation: 观测（图像、关节状态等）
            - actions: 专家动作标签

    Returns:
        (new_state, info): 更新后的训练状态和训练信息字典
        - info 包含 loss, grad_norm, param_norm
    """
    # 从 graphdef 和参数重建模型 [外部 Flax]
    # 将静态的模型结构定义（model_def）与动态的参数（params）合并
    # 还原成一个可执行的模型实例。这是 Flax NNX 库特有的方式，用于处理状态管理。
    model = nnx.merge(state.model_def, state.params)
    # 设置为训练模式（影响 dropout、batch norm 等）[外部 Flax]
    # 这会激活某些在推理时不需要的层
    # 例如 Dropout（随机丢弃神经元以防止过拟合）或 Batch Normalization 的统计量更新。
    model.train()

    @at.typecheck  # [内部] 类型检查装饰器
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        """损失函数：计算模型预测与专家动作的差异。

        Args:
            model: 模型实例 [内部 BaseModel 子类]
            rng: 随机数键
            observation: 观测数据
            actions: 专家动作标签

        Returns:
            标量损失值
        """
        # model.compute_loss [内部] 模型定义的损失计算
        # train=True 表示训练模式（可能使用 dropout）
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        # jnp.mean [外部 JAX] 计算平均损失
        return jnp.mean(chunked_loss)

    # 使用当前步数折叠随机数键 [外部 JAX]
    # 根据当前训练步数生成一个新的随机数键
    # 确保每一步的随机行为（如 Dropout 掩码）都是不同的,同时保持可重现性
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # 创建差分状态，只对可训练参数计算梯度
    # nnx.DiffState [外部 Flax] 
    # 它告诉 JAX 我们只对符合 config.trainable_filter 规则的参数（即可训练参数）计算梯度
    # 而冻结其他参数（如 Batch Norm 的统计量或嵌入层）。
    diff_state = nnx.DiffState(0, config.trainable_filter)

    # 计算损失和梯度 [外部 Flax]
    # nnx.value_and_grad是JAX 的核心功能。
    # 它同时执行前向传播（计算 loss）和反向传播（计算 grads）。 
    # argnums=diff_state 指定只对可训练参数求导
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # 过滤出可训练参数 [外部 Flax]
    # 只提取需要更新的参数。
    params = state.params.filter(config.trainable_filter)

    # 优化器更新 [外部 Optax]
    # tx.update: 根据梯度计算参数更新
    # 调用优化器（如 AdamW, SGD 等）的计算逻辑。
    # 它根据梯度和当前的优化器状态（如动量），计算出参数的更新量（updates）
    # 并返回新的优化器状态。
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)

    # 应用参数更新 [外部 Optax]
    # optax.apply_updates: params = params + updates
    new_params = optax.apply_updates(params, updates)

    # 就地更新模型参数 [外部 Flax]
    nnx.update(model, new_params)
    # 提取更新后的模型状态 [外部 Flax]
    new_params = nnx.state(model)

    # 创建新的训练状态 [标准库 dataclasses]
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)

    # 更新 EMA (Exponential Moving Average) 参数
    # EMA 用于推理时获得更稳定的预测
    if state.ema_decay is not None:
        # EMA 公式: new_ema = decay * old_ema + (1 - decay) * new_param
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params
            ),
        )

    # 过滤出核参数（排除 bias、scale 等）
    # 用于计算参数范数，监控权重大小
    kernel_params = nnx.state(
        model,
        nnx.All(  # [外部 Flax] 组合多个过滤条件
            nnx.Param,  # 只选择参数
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),  # [内部] 排除这些参数
            lambda _, x: x.value.ndim > 1,  # 只选择多维参数（核）
        ),
    )

    # 构造训练信息字典
    # loss: 当前的训练损失，用于判断模型是否收敛。
    # grad_norm: 梯度的全局范数。如果该值过大（梯度爆炸）或过小（梯度消失），说明训练不稳定。
    # param_norm: 参数的大小，用于监控权重是否在合理范围内。
    info = {
        "loss": loss,                              # 训练损失
        "grad_norm": optax.global_norm(grads),     # 梯度范数 [外部 Optax]
        "param_norm": optax.global_norm(kernel_params),  # 参数范数 [外部 Optax]
    }
    # 返回更新后的 new_state 和包含监控信息的 info 字典
    # 供外层训练循环记录日志或进行下一步迭代。
    return new_state, info


# ============================================================================
# 第六部分：验证步骤
# ============================================================================

@at.typecheck  # [内部] 类型检查装饰器
def validation_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """执行验证步骤：计算验证集上的损失，不更新参数。

    Args:
        config: 训练配置 [内部]
        rng: 随机数键 [JAX PRNGKey]
        state: 当前训练状态 [内部 TrainState]
        batch: 验证批次 (observation, actions) [内部类型]

    Returns:
        {"val_loss": 验证损失}
    """
    # 使用 EMA 参数（如果可用），否则使用当前参数
    # EMA 参数通常提供更稳定的评估结果
    params = state.ema_params if state.ema_params is not None else state.params

    # 从 graphdef 和参数重建模型 [外部 Flax]
    # 将模型定义（graphdef）和参数合并成完整的模型实例
    model = nnx.merge(state.model_def, params)
    # 设置为评估模式（禁用 dropout）[外部 Flax]
    # 禁用 Dropout: 评估时不使用随机丢弃神经元
    # 禁用 BatchNorm 更新: 使用运行统计量而非批次统计量
    # 其他训练特定的行为也会被禁用
    model.eval()

    @at.typecheck  # [内部] 类型检查装饰器
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        """验证损失函数：计算模型预测与标签的差异。

        注意：train=False 禁用 dropout 等训练特定行为
        """
        # train=False 表示评估模式 [内部]
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
        return jnp.mean(chunked_loss)

    # 使用当前步数折叠随机数键 [外部 JAX]
    val_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # 计算验证损失（不计算梯度）
    loss = loss_fn(model, val_rng, observation, actions)

    return {"val_loss": loss}


# ============================================================================
# 第七部分：主训练循环
# ============================================================================

def main(config: _config.TrainConfig):
    """主训练函数：设置环境并执行训练循环。

    Args:
        config: 训练配置对象 [内部] 通过命令行参数创建
    """
    # 初始化自定义日志格式 [内部]
    init_logging()
    # 记录运行主机 [标准库]
    # 记录当前运行节点的主机名，方便在集群环境中追踪任务。
    logging.info(f"Running on: {platform.node()}")

    # 验证批次大小可以被设备数量整除
    # 确保每个设备获得相同数量的数据
    # 这是分布式训练的基本要求，确保每个 GPU/TPU 分配到的数据量一致。
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # 设置 JAX 编译缓存目录 [外部 JAX]
    # 缓存编译结果以加速后续启动
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 创建随机数键并分裂 [外部 JAX]
    # jax.random.key 创建主随机数键
    # 通过 split 生成用于训练循环和模型初始化的独立子键，保证实验的可复现性。
    rng = jax.random.key(config.seed)
    # train_rng   用于训练循环中的 dropout / noise
    # init_rng    用于模型参数初始化
    train_rng, init_rng = jax.random.split(rng)

    # 创建设备网格 [内部] 用于分布式训练
    mesh = sharding.make_mesh(config.fsdp_devices)

    # 创建数据分片策略 [外部 JAX]
    # 数据沿 "data" 轴分片到不同设备
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # 创建复制分片策略 [外部 JAX]
    # 定义某些参数（如标量损失值、全局步数）在所有设备上保持副本一致。
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 初始化检查点管理器 [内部]
    # 处理模型的保存、恢复、覆盖逻辑以及保留最近多少个检查点的策略。
    # resuming 变量标识当前是否为断点续训。
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,      # 保留多少个检查点
        overwrite=config.overwrite,          # 是否覆盖已有检查点
        resume=config.resume,                # 是否恢复训练
        save_full_state=config.save_full_state,  # 是否保存完整状态（包括优化器）
    )

    # 初始化 wandb [外部] 实验追踪
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 检查是否需要跳过归一化统计（用于 HDF5 等自定义数据集）
    data_config_for_check = config.data.create(config.assets_dirs, config.model)
    skip_norm_stats = data_config_for_check.norm_stats is None

    # 创建训练数据加载器 [内部]
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        split="train" if config.valid else None,
    )
    data_iter = iter(data_loader) #获取对象的迭代器
    batch = next(data_iter)       #获取迭代器的下一个元素

    # 记录数据批次信息（用于调试）
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 创建验证数据加载器（如果启用验证）
    # 使用 shuffle 以获得更好的 IID 采样
    val_iter = None
    if config.valid:
        val_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            split="val",
            training=False,
        )
        val_iter = iter(val_loader)  # 创建持久化的验证迭代器
        logging.info(f"Initialized validation data loader")

    # 记录第一批的相机图像到 wandb [外部]
    # 用于直观检查数据预处理是否正确。
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # 初始化训练状态 [内部]
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)

    # 等待训练状态就位 [外部 JAX]
    jax.block_until_ready(train_state)

    # 记录训练状态信息
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 如果是恢复训练，从检查点加载状态 [内部]
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # JIT 编译训练步骤 [外部 JAX]
    # jax.jit 对 train_step 进行即时编译，将其转化为高效的 XLA 计算图。
    # functools.partial [标准库] 固定 config 参数
    # in_shardings/out_shardings [外部 JAX] 
    # 明确指定输入输出在设备间的分布方式，减少不必要的跨设备通信。
    # donate_argnums=(1,) [外部 JAX] 
    # 告诉 JAX 在计算完成后可以释放旧的 train_state 内存，避免在更新参数时产生巨大的内存拷贝开销
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # JIT 编译验证步骤（如果启用） [外部 JAX]
    pval_step = None
    if config.valid:
        pval_step = jax.jit(
            functools.partial(validation_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    # 创建进度条 [外部 tqdm]
    # 使用 tqdm 显示训练进度，支持动态列宽和断点续训后的起始位置调整。
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,  # 动态调整列宽
    )

    # ============================================================================
    # 训练循环
    # ============================================================================
    infos = []  # 累积训练信息

    for step in pbar:
        # 在设备网格上下文中执行训练步骤 [内部]
        with sharding.set_mesh(mesh):
            # ptrain_step: JIT 编译的训练步骤
            train_state, info = ptrain_step(train_rng, train_state, batch)

        # 累积训练信息（用于平均）
        infos.append(info)

        # 定期记录日志
        if step % config.log_interval == 0:
            # 堆叠累积的信息并计算平均值 [外部 Flax]
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            # 运行验证（如果启用）
            if config.valid and val_iter is not None:
                val_losses = []
                num_val_batches = 10  # 每次验证使用 10 个批次
                for _ in range(num_val_batches):
                    val_batch = next(val_iter)  # 使用持久化迭代器
                    with sharding.set_mesh(mesh):
                        val_info = pval_step(train_rng, train_state, val_batch)
                    val_losses.append(val_info["val_loss"])

                # 计算平均验证损失
                if val_losses:
                    reduced_info["val_loss"] = float(jax.device_get(jnp.mean(jnp.array(val_losses))))

            # 格式化并打印信息
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")

            # 记录到 wandb [外部]
            wandb.log(reduced_info, step=step)
            infos = []  # 清空累积的信息

        # 获取下一批数据
        batch = next(data_iter)

        # 定期保存检查点 [内部]
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                config,
                step,
                save_full_state=config.save_full_state,
            )

    # 等待检查点管理器完成所有异步保存操作 [内部]
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    # 通过命令行接口创建配置并启动训练 [内部]
    # _config.cli() 解析命令行参数并创建 TrainConfig 对象
    main(_config.cli())
