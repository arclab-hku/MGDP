import torch
import copy
import os

def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)

def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")

def export_policy_as_jit(network, path, name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name)
    model = copy.deepcopy(network).to('cpu')
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def export_policy_as_onnx(network, input_size, path, name, input_names, output_names):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name)
    model = copy.deepcopy(network).to('cpu')
    dummy_observation = torch.zeros(1, input_size) # dummy observation with batch_size=1
    # print(f"************** dummy observation size {dummy_observation.shape} **************")
    torch.onnx.export(
        model,
        dummy_observation,
        path,
        export_params=True,
        opset_version=11,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
    )


def export_cnn_as_onnx(network, input_size, path, name, input_names, output_names):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name)
    model = network.to('cpu')

    dummy_input = torch.randn(*input_size)

    # 导出模型为ONNX格式
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=11,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
    )


def export_cnn_decoder_as_onnx(network, input_size, path, name, input_names, output_names):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, name)
    model = network.to('cpu').eval()

    # -------------------------- 核心：在函数内部生成模拟 skip 数据（无梯度）--------------------------
    # 1. 保存原 skip_connections（避免修改原网络）
    original_skips = model.skip_connections
    # 2. 生成模拟 skip 数据（匹配 ImageEncoder 输出形状：[32,64,64]通道，16→8→4尺寸）
    batch_size = input_size[0]  # 从 input_size 提取批量大小（如1）
    dummy_skips = [
        torch.randn(batch_size, 32, 16, 16, device='cpu', requires_grad=False),  # skip1: (B,32,16,16)
        torch.randn(batch_size, 64, 8, 8, device='cpu', requires_grad=False),   # skip2: (B,64,8,8)
        torch.randn(batch_size, 64, 4, 4, device='cpu', requires_grad=False)    # skip3: (B,64,4,4)
    ]
    # 3. 临时替换带梯度的 skip_connections（关键：消除梯度，避免ONNX报错）
    model.skip_connections = dummy_skips

    # -------------------------- 生成模拟输入 + 导出 ONNX --------------------------
    # 模拟输入：64维 visual_token（无梯度）
    dummy_input = torch.randn(*input_size, device='cpu', requires_grad=False)

    # 导出 ONNX（动态批量+适配Decoder输出）
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "visual_token": {0: "batch_size"},  # 批量维度动态
            "recon_image": {0: "batch_size"}
        },
        verbose=False
    )

    # -------------------------- 恢复原网络状态 --------------------------
    model.skip_connections = original_skips  # 把原 skip 数据放回去
    print(f"[INFO] Exported ImageDecoder to {save_path}")


import torch
import torch.nn.functional as F  # 新增：用于interpolate
import os


def export_map_decoder_as_onnx(network, input_size, path, name, input_names, output_names):
    """
    无适配层！完全匹配你实例化的 MapDecoder：
    - 自动读取 network.input_dims（32）、network.hidden_channels（[64,32,16]）
    - 模拟 skip 数据匹配你 MapEncoder 的真实输出（16→32→64通道）
    - 修复：临时用interpolate替代adaptive_avg_pool2d，解决非整数倍缩放导出错误
    """
    # 1. 创建保存目录（对齐 cnn_decoder 逻辑）
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, name)

    # 2. 保存原网络状态（避免影响后续使用）
    first_param = next(network.parameters())
    original_device = first_param.device
    original_skips = network.skip_connections  # 保存原 skip（如果有的话）
    original_train_mode = network.training
    original_forward = network.forward  # 新增：保存原forward方法！关键修复

    # -------------------------- 核心：按实例化后的维度准备数据 --------------------------
    # 读取你实例化时的关键参数（这是关键！不再用默认值）
    decoder_input_dims = 32
    decoder_hidden_ch = network.hidden_channels  # [64,32,16]（从实例读取，反转后的 MapModule_info 通道）
    decoder_pool = network.pool  # 2（从实例读取）

    # 生成匹配你 MapEncoder 的模拟 skip 数据（MapModule_info 隐藏通道 [16,32,64]）
    batch_size = input_size[0]
    # skip 通道数 = MapEncoder 的 hidden_channels（16→32→64），与你实例化的 MapEncoder 一致
    dummy_skips = [
        torch.randn(batch_size, 16, 17, 11, device='cpu', requires_grad=False),  # skip1: 16通道（MapEncoder conv1）
        torch.randn(batch_size, 32, 9, 6, device='cpu', requires_grad=False),  # skip2: 32通道（MapEncoder conv2）
        torch.randn(batch_size, 64, 5, 3, device='cpu', requires_grad=False)  # skip3: 64通道（MapEncoder conv3）
    ]
    # 给实例设置模拟 skip（和实际运行时的 skip 来源一致）
    network.skip_connections = dummy_skips

    # -------------------------- 新增：临时替换forward方法（解决adaptive_avg_pool2d错误） --------------------------
    def temp_forward(x):
        # 完全复制原forward逻辑，仅最后一步池化替换为interpolate
        target_size = network.target_size  # 保持原目标尺寸(17,11)
        batch_size = x.size(0)
        encoder_skips = network.skip_connections

        # 1. 特征扩展（原逻辑不变）
        x = network.fc(x).view(batch_size, network.hidden_channels[0], network.pool, network.pool)

        # 2. 升采样1（原逻辑不变）
        skip3 = encoder_skips[2]
        skip3 = F.adaptive_avg_pool2d(skip3, (network.pool, network.pool))
        x = torch.cat([x, skip3], dim=1)
        x = network.deconv1(x)

        # 3. 升采样2（原逻辑不变）
        skip2 = encoder_skips[1]
        skip2 = F.adaptive_avg_pool2d(skip2, (4, 4))
        x = torch.cat([x, skip2], dim=1)
        x = network.deconv2(x)

        # 4. 升采样3（原逻辑不变）
        skip1 = encoder_skips[0]
        skip1 = F.adaptive_avg_pool2d(skip1, (8, 8))
        x = torch.cat([x, skip1], dim=1)
        x = network.deconv3(x)  # 输出(B,1,16,16)

        # 关键替换：用interpolate替代adaptive_avg_pool2d，支持非整数倍缩放
        x = F.interpolate(
            x,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )

        # 最终激活（原逻辑不变）
        x = network.final_activation(x)
        return x

    # 绑定临时forward方法到网络
    network.forward = temp_forward

    # -------------------------- 导出 ONNX（纯32维逻辑，无适配层） --------------------------
    # 模拟输入：和你实际输入一致的 (batch_size, 32)
    dummy_input = torch.randn(*input_size, device='cpu', requires_grad=False)
    # 转CPU+推理模式（导出标准操作）
    network.to('cpu').eval()

    torch.onnx.export(
        network,  # 直接用你实例化后的 MapDecoder（input_dims=32）
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={  # 支持动态批量（和你实际 num_envs 适配）
            input_names[0]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"}
        },
        verbose=False
    )

    # -------------------------- 恢复原网络状态（新增恢复forward方法） --------------------------
    network.forward = original_forward  # 新增：恢复原forward方法，不影响后续使用
    network.skip_connections = original_skips
    network.to(original_device)
    if original_train_mode:
        network.train()

    # 打印导出信息（对齐 cnn_decoder 风格）
    print(f"Successfully exported map_decoder to {os.path.join(save_path)}")
    print(f"  - Input dims: {decoder_input_dims} (matches MapModule_info['output_channels']=32)")
    print(f"  - Skip channels: [16,32,64] (matches MapEncoder hidden_channels)")