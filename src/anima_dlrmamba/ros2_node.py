"""DLRMamba ROS2 node — publishes Detection2DArray from RGB+IR input."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

try:
    import rclpy
    from sensor_msgs.msg import Image
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
    from std_msgs.msg import Header

    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

from .config import load_config
from .models.model import DLRMambaDetector


class DLRMambaNode:
    """ROS2 node for DLRMamba inference on paired RGB/IR streams."""

    def __init__(self):
        if not HAS_ROS2:
            raise RuntimeError("ROS2 (rclpy) not available. Install ros-jazzy-desktop.")

        rclpy.init()
        self.node = rclpy.create_node("dlrmamba_node")

        config_path = os.getenv("DLRMAMBA_CONFIG", "configs/default.toml")
        checkpoint = os.getenv("DLRMAMBA_CHECKPOINT", "")

        self.cfg = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DLRMambaDetector(
            num_classes=self.cfg.model.num_classes,
            in_channels=self.cfg.model.in_channels,
            fusion_channels=self.cfg.model.fusion_channels,
            embed_dim=self.cfg.model.embed_dim,
            num_blocks=self.cfg.model.num_blocks,
            state_dim=self.cfg.model.state_dim,
            rank_ratio=self.cfg.model.rank_ratio,
        ).to(self.device)
        self.model.eval()

        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
            state = ckpt.get("model", ckpt)
            self.model.load_state_dict(state, strict=False)
            self.node.get_logger().info(f"Loaded checkpoint: {checkpoint}")

        self.rgb_msg = None
        self.ir_msg = None

        self.sub_rgb = self.node.create_subscription(
            Image, "/camera/rgb/image_raw", self._rgb_callback, 10
        )
        self.sub_ir = self.node.create_subscription(
            Image, "/camera/ir/image_raw", self._ir_callback, 10
        )
        self.pub_det = self.node.create_publisher(Detection2DArray, "/dlrmamba/detections", 10)

        self.node.get_logger().info("DLRMamba node ready")

    def _rgb_callback(self, msg: Image):
        self.rgb_msg = msg
        self._try_inference()

    def _ir_callback(self, msg: Image):
        self.ir_msg = msg
        self._try_inference()

    def _msg_to_tensor(self, msg: Image) -> torch.Tensor:
        """Convert ROS Image message to [3, H, W] tensor."""
        h, w = msg.height, msg.width
        if msg.encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if msg.encoding == "bgr8":
                arr = arr[:, :, ::-1].copy()
        elif msg.encoding == "mono8":
            mono = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
            arr = np.stack([mono, mono, mono], axis=-1)
        else:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)

        t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0),
            size=(self.cfg.data.image_size, self.cfg.data.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return t

    @torch.no_grad()
    def _try_inference(self):
        if self.rgb_msg is None or self.ir_msg is None:
            return

        rgb = self._msg_to_tensor(self.rgb_msg).to(self.device)
        ir = self._msg_to_tensor(self.ir_msg).to(self.device)
        sample = torch.stack([rgb, ir], dim=0).unsqueeze(0)

        out = self.model(sample)
        preds = self.model.decode(
            out,
            conf_threshold=self.cfg.infer.conf_threshold,
            topk=self.cfg.infer.topk,
        )[0]

        det_array = Detection2DArray()
        det_array.header = Header()
        det_array.header.stamp = self.node.get_clock().now().to_msg()
        det_array.header.frame_id = "camera"

        for p in preds:
            det = Detection2D()
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(p["class_id"])
            hyp.hypothesis.score = p["score"]
            det.results.append(hyp)
            det.bbox.center.position.x = p["bx"] * self.rgb_msg.width
            det.bbox.center.position.y = p["by"] * self.rgb_msg.height
            det.bbox.size_x = p["bw"] * self.rgb_msg.width
            det.bbox.size_y = p["bh"] * self.rgb_msg.height
            det_array.detections.append(det)

        self.pub_det.publish(det_array)
        self.rgb_msg = None
        self.ir_msg = None

    def spin(self):
        rclpy.spin(self.node)

    def shutdown(self):
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    if not HAS_ROS2:
        print("ROS2 not available. Install ros-jazzy-desktop to use the ROS2 node.")
        print("For API-only mode, use: uvicorn anima_dlrmamba.serve:app")
        return

    node = DLRMambaNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
