# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BPX 四足机器人自定义奖励/惩罚函数集合。

本模块为 manager-based RL 环境提供各类奖励项，主要服务于"趴卧起身"(recovery / stand-up)
及相关运动任务。所有函数都遵循 Isaac Lab 奖励函数的统一签名::

    def reward_fn(env: ManagerBasedRLEnv, *args, asset_cfg: SceneEntityCfg) -> torch.Tensor

并返回形状为 ``(num_envs,)`` 的张量，每个元素是对应环境当前步的奖励值
(正/负由外部 RewardManager 中的 weight 控制)。

设计思路概览:
    - 大多数"起身"相关的奖励都用 ``height_progress``(实际抬升进度, 0~1)做门控,
      避免机器人在趴地阶段白拿奖励, 或在还没抬起来时就受到站姿/防滑等约束;
    - 使用指数核 ``exp(-error^2 / std^2)`` 把误差映射到 (0, 1] 的稠密奖励,
      误差越小奖励越接近 1, 为稀疏的成功信号提供中间引导;
    - ``recovered_*`` 系列为二值(0/1)判定函数, 用于标记"起身成功"这一事件,
      也可配合 :func:`recovered_stability_with_contact` 生成成功后的稳定奖励。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚关节位置偏离目标值(L2 平方误差和)。

    常用于约束关节回到默认位置(target=0)或其他期望姿态。
    先把关节角包裹到 (-pi, pi], 避免多圈关节因角度绕回导致误差计算错误。

    参数:
        env: RL 环境实例。
        target: 期望的关节角(弧度), 所有被 ``asset_cfg.joint_ids`` 选中的关节共用该目标值。
        asset_cfg: 场景实体配置, 指定机器人名称及作用的关节索引。

    返回:
        形状 ``(num_envs,)`` 的张量, 值为各关节偏差的平方和。
    """
    # 提取关节所在的 Articulation 对象(显式标注类型以获得类型提示)
    asset: Articulation = env.scene[asset_cfg.name]
    # 将关节角包裹到 (-pi, pi), 消除角度绕回对误差的影响
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # 各关节偏差的平方和作为惩罚量
    return torch.sum(torch.square(joint_pos - target), dim=1)


def _left_right_joint_pair_errors(asset: Articulation, joint_ids) -> torch.Tensor:
    """Return mirrored pair errors for roll, pitch and knee joints."""
    offsets = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    if offsets.shape[1] != 12:
        raise ValueError("左右对称关节配置必须按策略顺序包含 12 个关节")
    return torch.stack(
        (
            offsets[:, 0] + offsets[:, 1],
            offsets[:, 2] + offsets[:, 3],
            offsets[:, 4] - offsets[:, 5],
            offsets[:, 6] - offsets[:, 7],
            offsets[:, 8] - offsets[:, 9],
            offsets[:, 10] - offsets[:, 11],
        ),
        dim=1,
    )


def left_right_joint_symmetry_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return RMS error from the mirrored left/right standing posture.

    The joint list must use policy order: four hip-roll joints, four hip-pitch
    joints and four knees, with front-left/front-right/hind-left/hind-right in
    each group. Hip-roll offsets have opposite signs under reflection; pitch and
    knee offsets have equal signs.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    errors = _left_right_joint_pair_errors(asset, asset_cfg.joint_ids)
    return torch.sqrt(torch.mean(torch.square(errors), dim=1))


def left_right_joint_symmetry_l2(
    env: ManagerBasedRLEnv,
    tolerance: float,
    scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize normalized mirrored joint-pair errors after a terrain deadband."""
    if tolerance < 0.0:
        raise ValueError("左右对称容差不能小于零")
    if scale <= 0.0:
        raise ValueError("左右对称误差归一化尺度必须大于零")
    asset: Articulation = env.scene[asset_cfg.name]
    errors = _left_right_joint_pair_errors(asset, asset_cfg.joint_ids)
    excess = torch.clamp(errors.abs() - tolerance, min=0.0)
    return torch.mean(torch.square(excess / scale), dim=1)


def whole_body_com_support_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the planar distance from the whole-body COM to the four-foot center."""
    asset: Articulation = env.scene[asset_cfg.name]
    if len(feet_cfg.body_ids) != 4:
        raise ValueError("支撑中心配置必须包含四个足端")

    body_com_xy = asset.data.body_com_pos_w[..., :2]
    masses = getattr(asset, "_bpx_default_mass_on_device", None)
    if masses is None or masses.device != body_com_xy.device or masses.dtype != body_com_xy.dtype:
        masses = asset.data.default_mass.to(device=body_com_xy.device, dtype=body_com_xy.dtype)
        asset._bpx_default_mass_on_device = masses
    whole_body_com_xy = torch.sum(body_com_xy * masses.unsqueeze(-1), dim=1)
    whole_body_com_xy /= masses.sum(dim=1, keepdim=True)

    feet_xy = asset.data.body_pos_w[:, feet_cfg.body_ids, :2]
    support_center_xy = feet_xy.mean(dim=1)
    return torch.linalg.vector_norm(whole_body_com_xy - support_center_xy, dim=1)


def whole_body_com_support_exp(
    env: ManagerBasedRLEnv,
    tolerance: float,
    std: float,
    asset_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward placing the whole-body COM above the center of the four feet.

    A small deadband preserves the posture freedom needed on rough terrain.  The
    exponential kernel gives this geometric objective a useful reward scale even
    when the position error is only a few centimeters.
    """
    if tolerance < 0.0:
        raise ValueError("质心居中容差不能小于零")
    if std <= 0.0:
        raise ValueError("质心居中奖励的 std 必须大于零")
    error = whole_body_com_support_error(env, asset_cfg, feet_cfg)
    excess = torch.clamp(error - tolerance, min=0.0)
    return torch.exp(-torch.square(excess / std))


def upright_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """用指数核奖励机身朝上。

    不能只使用 projected gravity 的 x/y 分量，因为完全倒置时这两个分量也可能为零。
    直立时机身坐标系中的重力 z 分量约为 -1，因此这里直接约束 z 分量接近 -1。

    参数:
        env: RL 环境实例。
        std: 指数核宽度, 越小对倾斜越敏感(奖励衰减越快)。
        asset_cfg: 场景实体配置, 默认为 "robot"。

    返回:
        形状 ``(num_envs,)`` 的张量, 完全直立时为 1, 倾斜后按指数衰减。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # projected_gravity_b 是世界系重力向量在机身坐标系下的投影;
    # 直立时其 z 分量为 -1, 偏离 -1 说明机身发生了倾斜
    gravity_z_error = asset.data.projected_gravity_b[:, 2] + 1.0
    # 指数核: 误差为 0 时奖励为 1, 误差越大奖励按 exp(-e^2/std^2) 衰减
    return torch.exp(-torch.square(gravity_z_error) / std**2)


def asymmetric_pitch_band_l2(
    env: ManagerBasedRLEnv,
    maximum_backward_pitch: float,
    maximum_forward_pitch: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """分别惩罚超出非对称容许区间的前倾和后仰。

    BPX 机体系 x 轴指向前方、y 轴指向左侧。按该坐标约定，机头向上后仰时
    ``projected_gravity_b[:, 0]`` 为负，机头向下前倾时为正。使用重力投影而不是
    欧拉角可避免姿态表示奇异，并允许前后方向设置不同的容许角度与惩罚强度。
    """
    if maximum_backward_pitch < 0.0 or maximum_backward_pitch >= math.pi / 2.0:
        raise ValueError("maximum_backward_pitch 必须位于 [0, pi/2) rad")
    if maximum_forward_pitch < 0.0 or maximum_forward_pitch >= math.pi / 2.0:
        raise ValueError("maximum_forward_pitch 必须位于 [0, pi/2) rad")
    asset: Articulation = env.scene[asset_cfg.name]
    maximum_backward_sine = math.sin(maximum_backward_pitch)
    maximum_forward_sine = math.sin(maximum_forward_pitch)
    gravity_x = asset.data.projected_gravity_b[:, 0]
    backward_excess = torch.clamp(
        -gravity_x - maximum_backward_sine,
        min=0.0,
    )
    forward_excess = torch.clamp(
        gravity_x - maximum_forward_sine,
        min=0.0,
    )
    return torch.square(backward_excess) + torch.square(forward_excess)


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """用指数核奖励机身高度接近目标值，作为 Init 起身任务的稠密高度信号。

    参数:
        env: RL 环境实例。
        target_height: 期望的机身(根部)世界系高度, 单位米。
        std: 指数核宽度, 控制对高度偏差的容忍程度。
        asset_cfg: 场景实体配置, 默认为 "robot"。

    返回:
        形状 ``(num_envs,)`` 的张量, 高度误差为零时为 1, 否则按指数衰减。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # root_pos_w 为机身根部在世界系下的位置, 取 z 分量与目标高度作差
    height_error = asset.data.root_pos_w[:, 2] - target_height
    # 指数核映射到 (0, 1], 为起身过程提供稠密的高度引导信号
    return torch.exp(-torch.square(height_error) / std**2)


def base_upward_velocity_limit_l2(
    env: ManagerBasedRLEnv,
    maximum_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚超过上限的机身世界系向上速度。

    只惩罚正向超限量，不要求机身跟踪固定速度。因此低于上限的缓慢起身不受影响，
    静止或向下运动也不会因为偏离某个正速度目标而额外受罚。

    参数:
        env: RL 环境实例。
        maximum_velocity: 允许的最大世界系向上速度, 单位 m/s。
        asset_cfg: 场景实体配置, 默认为 "robot"。

    返回:
        形状 ``(num_envs,)`` 的张量, 为向上速度超限量的平方。
    """
    if not math.isfinite(maximum_velocity) or maximum_velocity < 0.0:
        raise ValueError("maximum_velocity 必须是非负有限值")
    asset: Articulation = env.scene[asset_cfg.name]
    upward_velocity = asset.data.root_lin_vel_w[:, 2]
    velocity_excess = torch.clamp(upward_velocity - maximum_velocity, min=0.0)
    return torch.square(velocity_excess)


def joint_torque_limit_l2(
    env: ManagerBasedRLEnv,
    maximum_torque: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚绝对值超过上限的关节施加力矩。

    阈值以内不产生惩罚；超过阈值后按超限量的平方快速增大。该项是训练期软约束，
    不能替代执行器或真实电机驱动器中的硬限流、限矩保护。

    参数:
        env: RL 环境实例。
        maximum_torque: 允许的最大关节力矩绝对值, 单位 N·m。
        asset_cfg: 场景实体配置, 指定参与评估的关节。

    返回:
        形状 ``(num_envs,)`` 的张量, 为所有关节力矩超限量的平方和。
    """
    if not math.isfinite(maximum_torque) or maximum_torque <= 0.0:
        raise ValueError("maximum_torque 必须是正有限值")
    asset: Articulation = env.scene[asset_cfg.name]
    joint_torque = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    torque_excess = torch.clamp(joint_torque - maximum_torque, min=0.0)
    return torch.sum(torch.square(torque_excess), dim=1)


def upright_height_progress(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    orientation_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """用实际抬升进度门控朝向奖励，避免趴平不动也获得满额直立奖励。

    单独的直立奖励存在漏洞: 机器人趴在地上一动不动时机身仍然朝上, 照样能拿满奖励。
    这里把直立评分乘以 [0, 1] 的抬升进度因子, 只有真正把身体抬起来才逐渐解锁直立收益。

    参数:
        env: RL 环境实例。
        start_height: 起身开始(趴卧)时的典型机身高度, 对应进度 0。
        target_height: 起身完成后的目标高度, 对应进度 1。
        orientation_std: 直立评分的标准差。
        asset_cfg: 场景实体配置, 默认为 "robot"。

    返回:
        形状 ``(num_envs,)`` 的张量, 抬升进度与直立评分的乘积。
    """
    if target_height <= start_height:
        raise ValueError("target_height 必须大于 start_height")
    if orientation_std <= 0.0:
        raise ValueError("orientation_std 必须大于零")
    asset: Articulation = env.scene[asset_cfg.name]
    # 抬升进度: 当前高度在 [start_height, target_height] 区间内的归一化位置, 截断到 [0, 1]
    height_progress = torch.clamp(
        (asset.data.root_pos_w[:, 2] - start_height) / (target_height - start_height),
        min=0.0,
        max=1.0,
    )
    # 重力 z 分量误差: 直立时 projected_gravity_b[:, 2] == -1(误差 0),
    # 完全倒置时为 +1(误差 2), 截断到 [0, 2] 防止数值噪声越界
    gravity_z_error = torch.clamp(
        asset.data.projected_gravity_b[:, 2] + 1.0,
        min=0.0,
        max=2.0,
    )
    # 指数核直立评分(注意这里未对误差取平方, 衰减相对平缓)
    upright = torch.exp(-gravity_z_error / orientation_std**2)

    # 抬得多且立得正才有满额奖励
    return height_progress * upright


def joint_posture_progress_exp(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """随实际抬升进度奖励关节接近默认站姿。

    趴卧初态本来就远离默认站姿，因此不能从复位后的第一步就强迫所有关节回零。
    用实际机身高度而非时间作为门控后，机器人抬得越高，四条腿恢复正常站姿的收益越大。

    参数:
        env: RL 环境实例。
        start_height: 趴卧时的典型高度, 对应门控 0。
        target_height: 目标站立高度, 对应门控 1。
        std: 关节误差指数核的标准差。
        asset_cfg: 场景实体配置, 指定机器人及参与评估的关节。

    返回:
        形状 ``(num_envs,)`` 的张量, 抬升进度与站姿评分的乘积。
    """
    if target_height <= start_height:
        raise ValueError("target_height 必须大于 start_height")
    if std <= 0.0:
        raise ValueError("std 必须大于零")
    asset: Articulation = env.scene[asset_cfg.name]
    # 抬升进度门控, 含义同 upright_height_progress
    height_progress = torch.clamp(
        (asset.data.root_pos_w[:, 2] - start_height) / (target_height - start_height),
        min=0.0,
        max=1.0,
    )
    # 关节角相对默认站姿的偏差, 包裹到 (-pi, pi) 防止多圈绕回
    joint_error = wrap_to_pi(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    # 对所有关节取平均平方误差(而非求和), 使奖励尺度不随关节数量变化
    posture = torch.exp(-torch.mean(torch.square(joint_error), dim=1) / std**2)
    # 站姿收益随抬升进度逐步解锁
    return height_progress * posture


def _front_feet_lateral_width(asset: Articulation, front_feet_cfg: SceneEntityCfg) -> torch.Tensor:
    """返回左前足到右前足的机体系有符号横向距离。

    作为多个前足宽度奖励的公共工具函数。机体系 y 轴通常指向机身左侧,
    因此正值表示左右前足正常分开, 负值表示前足交叉。

    参数:
        asset: 机器人 Articulation 对象。
        front_feet_cfg: 场景实体配置, 必须按 (左前, 右前) 顺序提供两个足端 body。

    返回:
        形状 ``(num_envs,)`` 的张量, 左右前足在机身坐标系 y 方向上的有符号间距(米)。
    """
    if len(front_feet_cfg.body_ids) != 2:
        raise ValueError("front_feet_cfg 必须按左前、右前顺序包含两个足端")
    # 两个前足足端在世界系下的位置
    left_position_w = asset.data.body_pos_w[:, front_feet_cfg.body_ids[0], :]
    right_position_w = asset.data.body_pos_w[:, front_feet_cfg.body_ids[1], :]
    # 先把世界系位置平移到机身根部, 再用四元数的逆旋转到机身坐标系
    # (quat_apply_inverse 即 q* ⊗ v ⊗ q, 与机体系姿态定义一致)
    left_position_b = quat_apply_inverse(
        asset.data.root_quat_w,
        left_position_w - asset.data.root_pos_w,
    )
    right_position_b = quat_apply_inverse(
        asset.data.root_quat_w,
        right_position_w - asset.data.root_pos_w,
    )
    # 左足 y 坐标减右足 y 坐标 = 有符号横向宽度
    return left_position_b[:, 1] - right_position_b[:, 1]


def front_feet_width_progress(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    minimum_width: float,
    front_feet_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """随抬升进度奖励前足在机体坐标系中保持足够横向间距。

    ``front_feet_cfg`` 必须按左前、右前顺序提供两个足端。使用有符号宽度而不是绝对值，
    因此前足交叉不会重新获得奖励。达到最低宽度后奖励饱和，不强迫策略跟踪固定轨迹。

    参数:
        env: RL 环境实例。
        start_height: 趴卧时的典型高度, 对应门控 0。
        target_height: 目标站立高度, 对应门控 1。
        minimum_width: 期望的前足最小横向间距(米), 达到后宽度评分饱和为 1。
        front_feet_cfg: 前足足端配置, 按 (左前, 右前) 顺序。
        asset_cfg: 机器人配置, 默认为 "robot"。

    返回:
        形状 ``(num_envs,)`` 的张量, 取值范围 [0, 1]。
    """
    if target_height <= start_height:
        raise ValueError("target_height 必须大于 start_height")
    if minimum_width <= 0.0:
        raise ValueError("minimum_width 必须大于零")
    asset: Articulation = env.scene[asset_cfg.name]
    # 有符号宽度归一化到 [0, 1]: 宽度不足时线性增长, 达到 minimum_width 后饱和;
    # 前足交叉(负宽度)时评分为 0, 且不会因 |宽度| 增大而重新得分
    signed_width = _front_feet_lateral_width(asset, front_feet_cfg)
    width_score = torch.clamp(signed_width / minimum_width, min=0.0, max=1.0)
    # 抬升进度门控, 含义同上
    height_progress = torch.clamp(
        (asset.data.root_pos_w[:, 2] - start_height) / (target_height - start_height),
        min=0.0,
        max=1.0,
    )
    return height_progress * width_score


def feet_support_progress(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励随实际抬升建立四足支撑，避免趴地时靠足端接触白拿奖励。

    参数:
        env: RL 环境实例。
        start_height: 趴卧时的典型高度, 对应门控 0。
        target_height: 目标站立高度, 对应门控 1。
        threshold: 判定足端接触的最小合力阈值(牛顿)。
        sensor_cfg: 接触传感器配置, 指定参与判定的足端 body。
        asset_cfg: 机器人配置, 默认为 "robot"。

    返回:
        形状 ``(num_envs,)`` 的张量, 抬升进度乘以四足接触比例(0~1)。
    """
    if target_height <= start_height:
        raise ValueError("target_height 必须大于 start_height")
    if threshold <= 0.0:
        raise ValueError("threshold 必须大于零")
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history 形状: (num_envs, history_length, num_bodies, 3)
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    # 取历史窗口内合力范数的最大值与阈值比较, 判定每个足端是否真实接触地面
    contacts = contact_forces.norm(dim=-1).max(dim=1)[0] > threshold
    # 处于接触状态的足端比例(0~1), 四足全部落地时为 1
    contact_fraction = contacts.float().mean(dim=1)
    # 抬升进度门控: 趴地时即使足端碰地也拿不到奖励
    height_progress = torch.clamp(
        (asset.data.root_pos_w[:, 2] - start_height) / (target_height - start_height),
        min=0.0,
        max=1.0,
    )
    return height_progress * contact_fraction


def feet_slide_progress(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """随抬升进度惩罚接触地面的足端滑动，保留趴姿早期必要的重排空间。

    参数:
        env: RL 环境实例。
        start_height: 趴卧时的典型高度, 对应门控 0。
        target_height: 目标站立高度, 对应门控 1。
        threshold: 判定足端接触的最小合力阈值(牛顿)。
        sensor_cfg: 接触传感器配置, 指定参与接触判定的足端。
        asset_cfg: 机器人配置, 其 body_ids 必须与 sensor_cfg 一一对应
            (数量与顺序均相同), 用于读取足端速度。

    返回:
        形状 ``(num_envs,)`` 的张量, 接触足端平面速度之和乘以抬升进度
        (惩罚项, 外部通常配负权重)。
    """
    if target_height <= start_height:
        raise ValueError("target_height 必须大于 start_height")
    if threshold <= 0.0:
        raise ValueError("threshold 必须大于零")
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 接触判定: 历史窗口内最大合力范数超过阈值即认为该足端在地面上
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contacts = contact_forces.norm(dim=-1).max(dim=1)[0] > threshold
    # 足端在世界系下的水平(x-y)速度大小
    foot_planar_speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    # 接触掩码与足端速度必须逐足对应, 否则惩罚会算错对象
    if contacts.shape[1] != foot_planar_speed.shape[1]:
        raise ValueError("sensor_cfg 与 asset_cfg 必须按相同顺序包含相同数量的足端")
    # 只惩罚"接触中的足端"的水平速度: 接触为 True 记 1, 悬空足端的滑动不计入
    slide = torch.sum(foot_planar_speed * contacts, dim=1)
    # 抬升进度门控: 起身早期允许拖动双脚重排, 站起来之后才严格要求不打滑
    height_progress = torch.clamp(
        (asset.data.root_pos_w[:, 2] - start_height) / (target_height - start_height),
        min=0.0,
        max=1.0,
    )
    return height_progress * slide


def recovered_posture_with_contact(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_tilt: float,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    minimum_front_width: float,
    front_feet_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """高度、姿态、四足接触和前足宽度都合格时，判定为完成起身。

    在 :func:`recovered_posture`(高度 + 姿态)的基础上增加两项检查:
    四足同时接触地面、前足横向间距不低于 ``minimum_front_width``,
    从而避免"高度够了但前足交叉/部分悬空"的伪成功被判定为起身完成。

    参数:
        env: RL 环境实例。
        minimum_height: 判定起身完成的最小机身高度(米)。
        maximum_tilt: 判定 upright 的最大倾角(弧度)。
        threshold: 判定足端接触的最小合力阈值(牛顿)。
        sensor_cfg: 接触传感器配置, 覆盖四个足端。
        minimum_front_width: 前足最小横向间距(米)。
        front_feet_cfg: 前足足端配置, 按 (左前, 右前) 顺序。
        asset_cfg: 机器人配置, 默认为 "robot"。

    返回:
        形状 ``(num_envs,)`` 的 0/1 张量, 1 表示该环境已完成起身。
    """
    if threshold <= 0.0:
        raise ValueError("threshold 必须大于零")
    if minimum_front_width <= 0.0:
        raise ValueError("minimum_front_width 必须大于零")
    # 条件一: 高度与姿态检查(见 recovered_posture)
    recovered = recovered_posture(env, minimum_height, maximum_tilt, asset_cfg).bool()
    # 条件二: 四个足端在历史窗口内都存在真实接触
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    all_feet_contact = (contact_forces.norm(dim=-1).max(dim=1)[0] > threshold).all(dim=1)
    # 条件三: 前足有符号间距达标(交叉时为负值, 自然不达标)
    asset: Articulation = env.scene[asset_cfg.name]
    front_width_valid = _front_feet_lateral_width(asset, front_feet_cfg) >= minimum_front_width
    # 三个条件同时满足才输出 1
    return torch.logical_and(torch.logical_and(recovered, all_feet_contact), front_width_valid).float()


def recovered_stability_with_contact(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_tilt: float,
    linear_velocity_std: float,
    angular_velocity_std: float,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    minimum_front_width: float,
    front_feet_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """仅在四足完成起身后奖励低机身速度，不限制起身过程所需的运动。

    直接惩罚机身速度会阻碍起身过程中的大幅运动, 因此先用
    :func:`recovered_posture_with_contact` 得到 0/1 的完成标记,
    只对已完成起身的环境按机身速度打折, 未起身的环境奖励恒为 0。

    参数:
        env: RL 环境实例。
        minimum_height: 判定起身完成的最小机身高度(米)。
        maximum_tilt: 判定 upright 的最大倾角(弧度)。
        linear_velocity_std: 线速度误差的归一化标准差。
        angular_velocity_std: 角速度误差的归一化标准差。
        threshold: 判定足端接触的最小合力阈值(牛顿)。
        sensor_cfg: 接触传感器配置, 覆盖四个足端。
        minimum_front_width: 前足最小横向间距(米)。
        front_feet_cfg: 前足足端配置, 按 (左前, 右前) 顺序。
        asset_cfg: 机器人配置, 默认为 "robot"。

    返回:
        形状 ``(num_envs,)`` 的张量: 未完成起身为 0; 完成后为
        ``1 / (1 + 归一化运动量)``, 站得越稳越接近 1。
    """
    if linear_velocity_std <= 0.0 or angular_velocity_std <= 0.0:
        raise ValueError("速度标准差必须大于零")
    # 0/1 起身完成标记(高度 + 姿态 + 四足接触 + 前足宽度)
    recovered = recovered_posture_with_contact(
        env,
        minimum_height,
        maximum_tilt,
        threshold,
        sensor_cfg,
        minimum_front_width,
        front_feet_cfg,
        asset_cfg,
    )
    asset: Articulation = env.scene[asset_cfg.name]
    # 机身坐标系下的线速度/角速度平方和
    linear_error = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)
    angular_error = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    # 用各自的标准差归一化后相加, 使线速度与角速度在同一尺度上参与比较
    normalized_motion = (
        linear_error / linear_velocity_std**2 + angular_error / angular_velocity_std**2
    )
    # 双曲型衰减: 静止时奖励为 1, 运动量越大奖励越小; 未起身的环境乘 0 直接归零
    return recovered / (1.0 + normalized_motion)


def action_rate_l2_after_time(env: ManagerBasedRLEnv, start_time_s: float) -> torch.Tensor:
    """跳过复位后的首次动作跳变，再惩罚相邻策略动作的变化率。

    回合开始时, 策略输出与上一回合末的动作(甚至是初始零向量)往往差异巨大,
    若立刻惩罚动作变化率, 会产生一笔与当前策略行为无关的大惩罚。
    因此在 ``start_time_s`` 之前把惩罚置零, 之后再按 L2 惩罚动作差分。

    参数:
        env: RL 环境实例。
        start_time_s: 回合开始后经过多少秒才计入惩罚。

    返回:
        形状 ``(num_envs,)`` 的张量, 动作差分平方和乘以时间掩码。
    """
    if start_time_s < 0.0:
        raise ValueError("start_time_s 不能小于零")
    # 当前回合已流逝时间: 步数缓冲 × 仿真步长
    elapsed_s = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    # 相邻两步策略动作的差分
    action_delta = env.action_manager.action - env.action_manager.prev_action
    # (elapsed >= start_time_s) 为布尔掩码, 转成浮点后逐环境相乘,
    # 早于起始时间的环境惩罚为 0
    return torch.sum(torch.square(action_delta), dim=1) * (elapsed_s >= start_time_s).float()


def recovered_posture(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_tilt: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机器人达到足够高度且方向朝上时，返回成功站立奖励。

    参数:
        env: RL 环境实例。
        minimum_height: 判定站立的最小机身高度(米)。
        maximum_tilt: 判定 upright 的最大倾角(弧度), 由重力方向反推。
        asset_cfg: 场景实体配置, 默认为 "robot"。

    返回:
        形状 ``(num_envs,)`` 的 0/1 张量, 同时满足高度与姿态条件时为 1。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # projected_gravity_b[:, 2] 的相反数即"向上程度"的余弦:
    # 直立时为 +1, 水平趴倒时为 0, 完全倒置时为 -1;
    # 截断到 [-1, 1] 防止数值误差使 acos 输入越界
    upright_cosine = torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)
    # 由余弦反求倾角(弧度), 0 表示完全直立
    tilt = torch.acos(upright_cosine)
    # 高度达标: 机身世界系 z 坐标不低于最小高度
    high_enough = asset.data.root_pos_w[:, 2] >= minimum_height
    # 姿态达标: 倾角不超过最大允许倾角
    upright_enough = tilt <= maximum_tilt
    # 两个条件同时满足 → 1, 否则 0
    return torch.logical_and(high_enough, upright_enough).float()
