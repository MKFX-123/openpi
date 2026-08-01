"""UMI (XRZero-G0 style) dual-arm symmetric policy transforms.

Ported from fastumi2jet/code/xv_dual_policy.py, adapted for the
pick_and_place umi-v260729 LeRobot dataset produced by
convert_umi_pickplace_to_lerobot.py.

Dataset fields consumed (per frame):
  - face_view / left_wrist_view / right_wrist_view  (uint8 HWC)
  - follow_left_pos(3) / follow_left_rotvec(3) / follow_left_gripper(1)
  - follow_right_pos(3) / follow_right_rotvec(3) / follow_right_gripper(1)
  - demo_start_pose_left(6) / demo_start_pose_right(6)   # first-frame pos+rotvec
  - left_action(7) / right_action(7)   # NEXT-frame ABSOLUTE pose+gripper (pos+rotvec+g)
  - task (str)

Action representation (same as fastumi2jet):
  - state  (12,): left/right current pose RELATIVE to demo-start, rot6d only (6+6).
  - actions (H,20): per hand  inv(T_cur) @ T_target  -> pos3 + rot6d6 + gripper1 = 10
                     left(10) + right(10) = 20.
  - The relative transform is frame-invariant under a constant map<->base transform,
    so the per-boot pico map origin and the map<->base rotation both cancel out.
    At deploy time the robot anchors to its own get_end_pose (base frame).

Differences from fastumi2jet:
  - No master/slave split; symmetric bimanual (no master data in umi-v260729).
  - Field names follow the umi-v260729 schema (follow_*_pos/rotvec/gripper).
  - Gripper /88 scaling removed (our gripper is in radians; norm_stats handles it).
"""

import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.policies.pose_util import (
    pose6_to_mat,
    mat_to_pose10d,   # NOTE: returns 9D (pos3 + rot6d6), despite the name
    mat_to_pose6,     # returns 6D (pos3 + rotvec3)
    pose10d_to_mat,   # NOTE: accepts 9D pose (pos3 + rot6d6)
)
from openpi.policies.pose_repr_util import convert_pose_mat_rep
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UmiInputs(transforms.DataTransformFn):
    """Training + inference input transform for the UMI dual-arm policy."""

    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        # --------------------------
        # 1) Images
        # --------------------------
        face_img = _parse_image(data["face_view"])
        left_img = _parse_image(data["left_wrist_view"])
        right_img = _parse_image(data["right_wrist_view"])

        inputs = {
            "image": {
                "base_0_rgb": face_img,
                "left_wrist_0_rgb": left_img,
                "right_wrist_0_rgb": right_img,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # --------------------------
        # 2) Low-dim state (current pose relative to demo start, rot6d only)
        # --------------------------
        l_pos = np.asarray(data["follow_left_pos"], np.float32)
        l_rot = np.asarray(data["follow_left_rotvec"], np.float32)
        r_pos = np.asarray(data["follow_right_pos"], np.float32)
        r_rot = np.asarray(data["follow_right_rotvec"], np.float32)

        l_cur_mat = pose6_to_mat(np.concatenate([l_pos, l_rot], axis=-1))  # (4,4)
        r_cur_mat = pose6_to_mat(np.concatenate([r_pos, r_rot], axis=-1))  # (4,4)

        l_start = np.asarray(data["demo_start_pose_left"], np.float32)
        r_start = np.asarray(data["demo_start_pose_right"], np.float32)
        l_start_mat = pose6_to_mat(l_start)
        r_start_mat = pose6_to_mat(r_start)

        l_rel_mat = convert_pose_mat_rep(l_cur_mat, l_start_mat, pose_rep="relative", backward=False)
        r_rel_mat = convert_pose_mat_rep(r_cur_mat, r_start_mat, pose_rep="relative", backward=False)

        # mat_to_pose10d returns 9D (pos3 + rot6d6); we keep rot6d only for the state.
        l_rel_rot6 = mat_to_pose10d(l_rel_mat)[3:]   # (6,)
        r_rel_rot6 = mat_to_pose10d(r_rel_mat)[3:]   # (6,)
        state12 = np.concatenate([l_rel_rot6, r_rel_rot6], axis=-1).astype(np.float32)
        inputs["state"] = state12  # model_transforms will pad to action_dim

        # --------------------------
        # 3) Action chunk: dataset raw actions are ABSOLUTE next-state per hand:
        #    left_action/right_action: [pos3, rotvec3, gripper1] (7D)
        #    Convert to RELATIVE-to-current: inv(T_cur) @ T_target -> pos3 + rot6d6 + gripper = 10D/hand.
        #    Final model actions: concat(left10, right10) -> (H, 20).
        # --------------------------
        raw_left = data.get("left_action", None)
        raw_right = data.get("right_action", None)

        if raw_left is not None and raw_right is not None:
            raw_left = np.asarray(raw_left, np.float32)
            raw_right = np.asarray(raw_right, np.float32)

            if raw_left.ndim == 1:
                raw_left = raw_left[None, :]
            if raw_right.ndim == 1:
                raw_right = raw_right[None, :]

            pad_mask = np.zeros((self.action_horizon,), dtype=np.bool_)
            T = min(raw_left.shape[0], raw_right.shape[0])
            if T < self.action_horizon:
                pad_mask[T:] = True
                raw_left = np.concatenate(
                    [raw_left[:T], np.repeat(raw_left[T - 1:T], self.action_horizon - T, axis=0)], axis=0
                )
                raw_right = np.concatenate(
                    [raw_right[:T], np.repeat(raw_right[T - 1:T], self.action_horizon - T, axis=0)], axis=0
                )
            else:
                raw_left = raw_left[: self.action_horizon]
                raw_right = raw_right[: self.action_horizon]

            # absolute target mats from raw next-state (pos + rotvec)
            left_tgt_mat = pose6_to_mat(np.concatenate([raw_left[:, :3], raw_left[:, 3:6]], axis=-1))   # (H,4,4)
            right_tgt_mat = pose6_to_mat(np.concatenate([raw_right[:, :3], raw_right[:, 3:6]], axis=-1))  # (H,4,4)

            # absolute targets -> relative wrt current pose: inv(T_cur) @ T_tgt
            left_rel_mat = convert_pose_mat_rep(left_tgt_mat, l_cur_mat, pose_rep="relative", backward=False)
            right_rel_mat = convert_pose_mat_rep(right_tgt_mat, r_cur_mat, pose_rep="relative", backward=False)

            # mat -> 9D (pos3 + rot6d6)
            left_pose9 = mat_to_pose10d(left_rel_mat).astype(np.float32)    # (H,9)
            right_pose9 = mat_to_pose10d(right_rel_mat).astype(np.float32)  # (H,9)

            left_grip = raw_left[:, 6:7].astype(np.float32)    # (H,1)
            right_grip = raw_right[:, 6:7].astype(np.float32)  # (H,1)

            left_act10 = np.concatenate([left_pose9, left_grip], axis=-1)    # (H,10)
            right_act10 = np.concatenate([right_pose9, right_grip], axis=-1)  # (H,10)

            inputs["actions"] = np.concatenate([left_act10, right_act10], axis=-1).astype(np.float32)  # (H,20)
            # actions_is_pad: True for padded (repeated-last) action steps.
            inputs["actions_is_pad"] = pad_mask

        # --------------------------
        # 4) Prompt
        # --------------------------
        if "prompt" in data:
            inputs["prompt"] = str(data["prompt"])
        elif "task" in data:
            inputs["prompt"] = str(data["task"])

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOutputs(transforms.DataTransformFn):
    """Inference-only transform.

    Model output: actions (H, 20):
      left:  pose9d(9) + gripper(1)  => 10
      right: pose9d(9) + gripper(1)  => 10

    Convert per-hand pose9d -> pose6 (pos3 + rotvec3) using pose10d_to_mat + mat_to_pose6,
    then append gripper => 7D per hand. No gripper scaling (gripper handled by norm_stats).

    Returns:
      - actions: (H, 14) = concat(left7, right7)
      - left_actions: (H, 7)
      - right_actions: (H, 7)
    """

    def __call__(self, data: dict) -> dict:
        act = np.asarray(data["actions"], dtype=np.float32)  # (H,20) or (20,)
        if act.ndim == 1:
            act = act[None, :]
        if act.shape[-1] != 20:
            raise ValueError(f"Expected model actions last-dim=20, got {act.shape}")

        l = act[..., :10]    # (H,10)
        r = act[..., 10:20]  # (H,10)

        l_pose9 = l[..., :9]
        l_grip = l[..., 9:10]
        r_pose9 = r[..., :9]
        r_grip = r[..., 9:10]

        # pose9 (relative transform) -> mat -> pose6 (pos3 + rotvec3)
        l_mat = pose10d_to_mat(l_pose9)
        r_mat = pose10d_to_mat(r_pose9)
        l_pose6 = mat_to_pose6(l_mat)   # (H,6)
        r_pose6 = mat_to_pose6(r_mat)   # (H,6)

        l7 = np.concatenate([l_pose6, l_grip], axis=-1).astype(np.float32)  # (H,7)
        r7 = np.concatenate([r_pose6, r_grip], axis=-1).astype(np.float32)  # (H,7)
        both14 = np.concatenate([l7, r7], axis=-1).astype(np.float32)       # (H,14)

        return {
            "actions": both14,
            "left_actions": l7,
            "right_actions": r7,
        }
