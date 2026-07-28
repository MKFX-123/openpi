import random
import dataclasses
import einops
import numpy as np
from typing import ClassVar

from openpi import transforms
from openpi.models import model as _model


def make_arx_example() -> dict:
    """Creates a random input example for the ARX policy."""
    return {
        "state": np.random.rand(14),
        "image": {
            "left_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "face_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Transform inputs for the ARX policy."""

    mode: str = "s2s"  # "s2s", "s2m", "sm2m", "sm2sm", "smp2smp"
    action_dim: int = 32
    model_type: _model.ModelType = _model.ModelType.PI0
    state_history_size: int = 0
    state_future_size: int = 0
    slave_state_dim: int = 14
    mask_history_slave_states: bool = False
    random_drop_master: float = 0.
    random_drop_history: float = 0.
    random_drop_future: float = 0.
    random_drop_label: float = 0.
    random_drop_label_global: bool = True
    random_pos_offset: float = 0.
    only_right_obs: bool = False
    mask_left_obs: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist_view", "face_view", "right_wrist_view")

    def __call__(self, data: dict) -> dict:
        state = data["state"]
        
        if state.ndim == 2:
            assert state.shape[0] == self.state_history_size + 1 + self.state_future_size
            state, master_mask = self._mask_states(state)
            state = transforms.pad_to_dim(state, self.action_dim)
            state[:, -1] = master_mask
        else:
            if random.random() < self.random_drop_master:
                state[self.slave_state_dim:] = state[:self.slave_state_dim]
                state = transforms.pad_to_dim(state, self.action_dim)
                state[-1] = 1.
            
        state = transforms.pad_to_dim(state, self.action_dim)
            
        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel].
            if img.shape[-1] != 3:
                output_image = einops.rearrange(img, "c h w -> h w c")
            else:
                output_image = img
            assert output_image.shape[-1] == 3, f"Image must have 3 channels, got {output_image.shape}."
            return output_image

        # Convert images to uint8 and rearrange to (H,W,C) format
        for key in self.EXPECTED_CAMERAS:
            assert key in data['images'].keys(), f"Images must contain {key}."
            data['images'][key] = convert_image(data['images'][key])

        inputs = {
            "image": {
                "base_0_rgb": data['images']['face_view'],
                "left_wrist_0_rgb": data['images']['left_wrist_view'],
                "right_wrist_0_rgb": data['images']['right_wrist_view'],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = data["actions_is_pad"]

        # random position offset augmentation
        if self.random_pos_offset > 0.:
            pos_offset = (np.random.rand(3) * 2 - 1.) * self.random_pos_offset
            inputs["state"][..., 7:10] += pos_offset
            inputs["actions"][..., 7:10] += pos_offset
            if self.mode in ["sm2m", "sm2sm", "smp2smp"]:
                inputs["state"][..., 21:24] += pos_offset
            if self.mode in ["sm2sm", "smp2smp"]:
                inputs["actions"][..., 21:24] += pos_offset
        
        if self.only_right_obs or self.mask_left_obs:
            inputs["image_mask"]["left_wrist_0_rgb"] = np.False_
            if self.slave_state_dim == 14:  # (left + right) x (pos + rot + gripper)
                inputs["state"][..., :7] = 0.
                if self.mode in ["sm2m", "sm2sm", "smp2smp"]:
                    inputs["state"][..., 14:21] = 0.
                if "actions" in inputs:
                    inputs["actions"][..., :7] = 0.
                    if self.mode in ["sm2sm", "smp2smp"]:
                        inputs["actions"][..., 14:21] = 0.
                        
            if self.only_right_obs:
                inputs["image_mask"]["base_0_rgb"] = np.False_

        if random.random() < self.random_drop_label:
            if self.random_drop_label_global or (inputs["state"][0, 28] != inputs["actions"][10, 28]):  # todo 定义 transition more robustly
                state[:, 28] = 0

        return inputs

    def _mask_states(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mask future slave states by copying current slave state."""
        state = np.asarray(state).copy()
        current_idx = self.state_history_size
        current_slave = state[current_idx, :self.slave_state_dim]
        current_state = state[current_idx]
        if state.shape[-1] == 32:
            master_mask = state[:, -1]
        else:
            master_mask = np.zeros((state.shape[0],), dtype=state.dtype)

        if self.state_future_size > 0:  # always mask future slave states
            state[current_idx + 1:, :self.slave_state_dim] = current_slave
        if self.mask_history_slave_states and self.state_history_size > 0:
            state[:current_idx, :self.slave_state_dim] = current_slave

        # Data augmentation: randomly drop history/future/master states
        if random.random() < self.random_drop_master:
            state[:, self.slave_state_dim:self.slave_state_dim*2] = current_slave
            master_mask[:] = 1.
        if random.random() < self.random_drop_history and self.state_history_size > 0:
            state[:current_idx] = current_state
            master_mask[:current_idx] = 1.
        if random.random() < self.random_drop_future and self.state_future_size > 0:
            mask_size = random.randint(1, self.state_future_size)
            state[-mask_size:] = state[-mask_size - 1]
            master_mask[-mask_size:] = 1.

        if self.mode == "smp2smp":
            state[:, 28] = state[current_idx, 28]
            
        return state, master_mask

@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Outputs for the ARX policy."""
    action_dim: int = 14
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :self.action_dim])
        return {"actions": actions}
