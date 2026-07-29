import dataclasses
import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class AdaptiveAccInputs(transforms.DataTransformFn):
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        factor = float(data.get("_speedup_factor", 1.0))
        actions = np.asarray(data["actions"])
        original_length = actions.shape[0]

        L = max(1, min(int(self.action_horizon * factor), original_length))

        if L != self.action_horizon:
            x_old = np.linspace(0, 1, L)
            x_new = np.linspace(0, 1, self.action_horizon)
            resampled = np.zeros((self.action_horizon, actions.shape[1]), dtype=actions.dtype)
            for dim in range(actions.shape[1]):
                resampled[:, dim] = np.interp(x_new, x_old, actions[:L, dim])
            data["actions"] = resampled

            if "actions_is_pad" in data:
                old_pad = np.asarray(data["actions_is_pad"])
                indices = np.linspace(0, min(L - 1, len(old_pad) - 1), self.action_horizon).astype(int)
                data["actions_is_pad"] = old_pad[indices]

        return data
