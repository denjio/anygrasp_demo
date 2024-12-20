import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import time

# 定义获取 RealSense 数据的函数
def get_realsense_data(pipeline, align):
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)  # 获取对齐帧，将深度框与颜色框对齐

    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    # 转换为 numpy 数组
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image


# 定义显示图像的函数
def show_images(color_image, depth_image):
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(color_image)
    plt.title("Color Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(depth_image, cmap='gray')
    plt.title("Depth Image")
    plt.axis("off")
    plt.show()

# 配置参数
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", default="/home/robot/Desktop/anygrasp_sdk/checkpoint/checkpoint_detection.tar",
                    help="Model checkpoint path")
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', default=True, action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()

cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

# 抓取跟踪函数
def demo():
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    vis = o3d.visualization.Visualizer()
    vis.create_window("Grasp Tracking", width=800, height=600)

    # 初始化跟踪状态
    target_grasp_ids = None
    # 启动管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    


    # 创建对齐对象与color流对齐
    align_to = rs.stream.color 
    align = rs.align(align_to)
    points = rs.points()
       
    try:
        # 持续接收并处理帧数据
        while True:
            # 获取 RealSense 数据
            color_image, depth_image = get_realsense_data(pipeline, align)
            if color_image is None or depth_image is None:
                print("Failed to capture data from RealSense")
                break

            print("Capture data from RealSense!")
            colors = np.asanyarray(color_image, dtype=np.float32) / 255.0
            depths = np.asanyarray(depth_image)
            colors = colors[..., ::-1]  # 将 BGR 转为 RGB
            # 显示图像
            # show_images(colors, depths)

            # 相机内参
            profile = pipeline.get_active_profile()
            intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            fx, fy = intr.fx, intr.fy
            cx, cy = intr.ppx, intr.ppy
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            # print(depths, depth_scale)
            
            # fx, fy = 654.14, 632.67
            # cx, cy = 632.67, 385.62
            # scale = 1000 

            # 点云生成
            xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            points_z = depths * depth_scale
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            # 使用遮罩过滤掉无效点
            mask = (points_z > 0) & (points_z < 0.85)
            points = np.stack([points_x, points_y, points_z], axis=-1)
           
            points = points[mask].astype(np.float32)
            colors = colors[mask].astype(np.float32)
        
            # 设置工作空间范围
            xmin = points[:, 0].min()
            xmax = points[:, 0].max()
            ymin = points[:, 1].min()
            ymax = points[:, 1].max()
            zmin = points[:, 2].min()
            zmax = points[:, 2].max()
            lims = [xmin, xmax, ymin, ymax, zmin, zmax]
            # 进行抓取检测
            print(points.shape, colors.shape)
            gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
            
            if len(gg) == 0:
                print('No Grasp detected after collision detection!')
                continue

            # 初次抓取选择
            if target_grasp_ids is None:
                gg = gg.nms().sort_by_score()
                target_grasp_ids = np.arange(min(len(gg), 10))  # 选择前10个抓取
                gg_pick = gg[target_grasp_ids]
                print("Initialized grasp tracking targets")

            # 获取最佳抓取点
            gg = gg.nms().sort_by_score()  # 排序抓取点，分数最高的在前
            best_grasp = gg[0]  # 获取分数最高的抓取点

            # 打印最佳抓取点的位置和分数
            print("Best grasp point location (translation):", best_grasp.translation)
            print("Best grasp score:", best_grasp.score)

            # 获取目标抓取点
            # 会反转图像
            # trans_mat = np.array([[1,0,0,0], [0,1,0,0], [0,0,-1,0], [0,0,0,1]])
            trans_mat = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg_pick.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)

            # 可视化更新
            vis.clear_geometries()
            vis.add_geometry(cloud)
            for gripper in grippers:
                vis.add_geometry(gripper)
            # vis.add_geometry(best_grasp)
            vis.poll_events()
            vis.update_renderer()
            
            # 模拟实时循环
            time.sleep(0.1)
        vis.destroy_window()
    finally:
        pipeline.stop()

if __name__ == '__main__':
    demo()
