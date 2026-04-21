import random
import dataclasses
import einops
import torch
import numpy as np
from typing import ClassVar

from openpi import transforms
from openpi.models import model as _model

# 探查 数据的结构和内容
def inspect(data):
    if isinstance(data, dict):
        return {key: inspect(value) for key, value in data.items()}  
    elif isinstance(data, list):
        return [inspect(item) for item in data]
    elif isinstance(data, np.ndarray):
        return f"ndarray(shape={data.shape}, dtype={data.dtype})"
    elif isinstance(data, torch.Tensor):
       return f"Tensor(shape={data.shape}, dtype={data.dtype})"
    else:
        return f"UnknownType({type(data)})"
    


@dataclasses.dataclass(frozen=True)
class VelocityAccInputs(transforms.DataTransformFn):
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            actions = data["actions"]
            original_length = actions.shape[0]

            # print(f"\n[VelocityAccInputs] 重采样前 data 结构:")
            # print(f"  actions.shape: {actions.shape}")
            # if "actions_is_pad" in data:
            #     print(f"  actions_is_pad.shape: {data['actions_is_pad'].shape}")
            # print(f"  目标 action_horizon: {self.action_horizon}")

            # 如果长度不匹配 action_horizon，进行线性插值重采样
            if original_length != self.action_horizon:
                # 对每个动作维度进行线性插值
                resampled_actions = np.zeros((self.action_horizon, actions.shape[1]), dtype=actions.dtype)
                x_old = np.linspace(0, 1, original_length)
                x_new = np.linspace(0, 1, self.action_horizon)

                for dim in range(actions.shape[1]):
                    resampled_actions[:, dim] = np.interp(x_new, x_old, actions[:, dim])

                data["actions"] = resampled_actions

                # 对 padding mask 进行最近邻插值
                if "actions_is_pad" in data:
                    old_pad = data["actions_is_pad"]
                    indices = np.linspace(0, original_length - 1, self.action_horizon).astype(int)
                    data["actions_is_pad"] = old_pad[indices]

            #     print(f"\n[VelocityAccInputs] 重采样后 data 结构:")
            #     print(f"  actions.shape: {data['actions'].shape}")
            #     if "actions_is_pad" in data:
            #         print(f"  actions_is_pad.shape: {data['actions_is_pad'].shape}")
            #     print(f"  重采样比例: {original_length} -> {self.action_horizon} (缩放因子: {self.action_horizon/original_length:.2f})")
            # else:
            #     print(f"\n[VelocityAccInputs] 长度匹配，无需重采样")

        return data
