import dataclasses
import logging
import struct
import socket
from collections import deque
from pathlib import Path
from typing import Literal

import os
import tyro
import json
import cv2
import numpy as np

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints

@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    policy_config: str = "throw_sm2m"
    policy_dir: str = "checkpoints/throw_sm2m/throw_0113_sm2m_h5f3/29999"
    policy_mode: Literal["s2s", "s2m", "sm2m", "sm2sm"] | None = None
    log_replay: bool = False
    state_history_size: int = None
    state_future_size: int = None
    state_step: int = None
    move_steps: int = 15
    only_right_arm: bool = False
    latency_step: int = None
    server_ip: str = None
    server_port: int = 57770
    skip_norm: bool = False

def _load_norm_stats(policy_config: str, policy_dir: str) -> dict | None:
    train_config = _config.get_config(policy_config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    return _checkpoints.load_norm_stats(Path(policy_dir) / "assets", data_config.asset_id)

def recv_all(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def read_img(conn):
    image_size = struct.unpack('<L', conn.recv(4))[0]
    image = recv_all(conn, image_size)
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def main(args: Args) -> None:
    # Auto-detect policy_mode from policy_dir if not specified
    if args.policy_mode is None:
        for mode in ['sm2sm', 'sm2m', 's2m', 's2s']:
            if mode in args.policy_dir.lower():
                args.policy_mode = mode
                logging.info(f"Auto-detected policy_mode from path: {args.policy_mode}")
                break
        if args.policy_mode is None:
            raise ValueError(f"Could not detect policy_mode from path: {args.policy_dir}. Please specify --policy-mode")
    
    # Load config params if not specified
    cfg = _config.get_config(args.policy_config)
    # 如果 skip_norm，在配置中设置 norm_stats=None 以禁用策略归一化变换
    if args.skip_norm:
        cfg = dataclasses.replace(cfg, norm_stats=None)
        logging.info("策略配置中已禁用归一化 (norm_stats=None)")
    if args.state_history_size is None:
        args.state_history_size = getattr(cfg.data, 'state_history_size', 0)
        logging.info(f"Using state_history_size from config: {args.state_history_size}")
    if args.state_future_size is None:
        args.state_future_size = getattr(cfg.data, 'state_future_size', 0)
        logging.info(f"Using state_future_size from config: {args.state_future_size}")
    if args.state_step is None:
        args.state_step = getattr(cfg.data, 'state_step', 1)
        logging.info(f"Using state_step from config: {args.state_step}")
    if args.latency_step is None:
        args.latency_step = args.state_future_size
        logging.info(f"Using latency_step equal to state_future_size: {args.latency_step}")
    if args.server_ip is None:
        args.server_ip = os.getenv("OPENPI_SERVER_IP", "0.0.0.0")
        logging.info(f"Using server_ip: {args.server_ip}")
    
    # Load policy
    logging.info(f"Loading policy from {args.policy_dir}")
    policy = _policy_config.create_trained_policy(cfg, args.policy_dir)
    # 如果未跳过归一化，则加载 norm stats（供脚本自己使用）
    norm_stats = None
    if not args.skip_norm:
        norm_stats = _load_norm_stats(args.policy_config, args.policy_dir)
        logging.info("已加载归一化统计信息供脚本使用")

    state_seq_len = args.state_history_size + 1 + args.state_future_size
    latency_len = args.state_history_size + 1 + args.latency_step
    master_queue = deque(maxlen=100)  # queue_len * 14
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(True) #设置通信是阻塞式
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.server_ip, args.server_port))
    sock.listen(1)
    print(f"Server is listening on {args.server_ip}:{args.server_port}")

    while True:
        conn, addr = sock.accept()
        print(f"Connection from {addr}")
        master_queue = deque(maxlen=100)
        try:
            while True:
                size_buf = conn.recv(4)
                if not size_buf:
                    raise ConnectionError("client disconnected")
                data_size = struct.unpack('<L', size_buf)[0]
                data = recv_all(conn, data_size)
                if data is None:
                    raise ConnectionError("client disconnected during payload")
                action_data = json.loads(data.decode('utf8'))

                left_agent_data = action_data['follow1_pos'] # (state_history_size + 1, 7)
                right_agent_data = action_data['follow2_pos'] # (state_history_size + 1, 7)

                image1 = read_img(conn)  # left
                image2 = read_img(conn)  # front
                image3 = read_img(conn)  # right

                h, w, c = np.array(image1).shape
                camera_front = np.array(image2).reshape(h, w, c)
                camera_left = np.array(image1).reshape(h, w, c)
                camera_right = np.array(image3).reshape(h, w, c)

                state = np.zeros((state_seq_len, 32), dtype=np.float32)
                slave_state = np.concatenate([left_agent_data, right_agent_data], axis=1) # (state_history_size + 1, 14)
                slave_state = np.concatenate([slave_state] + [slave_state[-1:]] * args.state_future_size)

                if not master_queue:
                    master_queue.extend([slave_state[-1]] * max(state_seq_len, latency_len))

                master_list = list(master_queue)[-latency_len:]
                if args.latency_step < args.state_future_size:  # inpainting mode
                    master_list = master_list + [master_list[-1]] * (args.state_future_size - args.latency_step)
                    state[args.latency_step - args.state_future_size:, -1] = 1.0
                else:  # naive async
                    master_list = master_list[:state_seq_len]
                master_state = np.array(master_list)

                if args.policy_mode in ["s2s", "s2m"]:
                    state[:, :14] = slave_state
                else:
                    state[:, :28] = np.concatenate([slave_state, master_state], axis=1)

                if args.only_right_arm:
                    if norm_stats is not None:
                        mean = np.asarray(norm_stats["state"].mean)
                        state[:, 0:7] = mean[..., 0:7]
                        if args.policy_mode in ["sm2m", "sm2sm"]:
                            state[:, 14:21] = mean[..., 14:21]
                    # 如果 norm_stats 为 None（skip_norm=True），状态维度保持其原始值

                obs = {
                    'images': {
                        'left_wrist_view': camera_left,
                        'face_view': camera_front,
                        'right_wrist_view': camera_right,
                    },
                    'prompt': '',
                    'state': state,
                }
                action_pred = policy.infer(obs)
                action_pred = action_pred['actions']
                if args.policy_mode == "sm2sm":
                    _, master_action = action_pred[:, :14], action_pred[:, 14:28]
                    action_pred = master_action

                action_pred = action_pred[args.latency_step:]
                action_pred = action_pred[:args.move_steps, ...]  # (move_steps, 14)
                action_pred = np.concatenate([[master_queue[-1]], action_pred])
                for action in action_pred[1:]:
                    master_queue.append(action)

                follow1_pos = action_pred[:, :7].tolist()
                follow2_pos = action_pred[:, 7:].tolist()

                data_dir ={
                    "follow1_pos":follow1_pos,
                    "follow2_pos":follow2_pos,
                }
                data_str = json.dumps(data_dir)
                data_bytes = data_str.encode('utf-8')
                conn.sendall(struct.pack('<L', len(data_bytes)))
                conn.sendall(data_bytes)
        except (ConnectionError, ConnectionResetError, BrokenPipeError) as exc:
            logging.info(f"Client disconnected: {exc}. Waiting for next connection.")
        finally:
            try:
                conn.close()
            except OSError:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
