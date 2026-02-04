import os
os.environ["OPENPI_DATA_HOME"] = "/mnt/public/xuyuanfan/.cache/openpi"
os.environ["HTTP_PROXY"] = "socks5://127.0.0.1:8899"
os.environ["HTTPS_PROXY"] = "socks5://127.0.0.1:8899"
os.environ["ALL_PROXY"] = "socks5://127.0.0.1:8899"
os.environ["all_proxy"] = "socks5://127.0.0.1:8899"
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

# config = config.get_config("pi0_fast_droid")
# config = config.get_config("pi05_base")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params")