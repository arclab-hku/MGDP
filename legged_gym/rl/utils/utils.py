import copy
import os

import torch
import torch.nn.functional as F


def split_and_pad_trajectories(tensor, dones):
    dones = dones.clone()
    dones[-1] = 1
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def tprint(*args):
    print("\r", end="")
    print(*args, end="")


def export_policy_as_jit(network, path, name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name)
    model = copy.deepcopy(network).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


def export_policy_as_onnx(network, input_size, path, name, input_names, output_names):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name)
    model = copy.deepcopy(network).to("cpu")
    dummy_observation = torch.zeros(1, input_size)
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
    model = network.to("cpu")
    dummy_input = torch.randn(*input_size)
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
    model = network.to("cpu").eval()
    original_skips = model.skip_connections
    batch_size = input_size[0]
    dummy_skips = [
        torch.randn(batch_size, 32, 16, 16, device="cpu", requires_grad=False),
        torch.randn(batch_size, 64, 8, 8, device="cpu", requires_grad=False),
        torch.randn(batch_size, 64, 4, 4, device="cpu", requires_grad=False),
    ]
    model.skip_connections = dummy_skips
    dummy_input = torch.randn(*input_size, device="cpu", requires_grad=False)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"visual_token": {0: "batch_size"}, "recon_image": {0: "batch_size"}},
        verbose=False,
    )
    model.skip_connections = original_skips
    print(f"[INFO] Exported ImageDecoder to {save_path}")


def export_map_decoder_as_onnx(network, input_size, path, name, input_names, output_names):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, name)
    first_param = next(network.parameters())
    original_device = first_param.device
    original_skips = network.skip_connections
    original_train_mode = network.training
    original_forward = network.forward
    decoder_input_dims = 32
    decoder_hidden_ch = network.hidden_channels
    decoder_pool = network.pool
    batch_size = input_size[0]
    dummy_skips = [
        torch.randn(batch_size, 16, 17, 11, device="cpu", requires_grad=False),
        torch.randn(batch_size, 32, 9, 6, device="cpu", requires_grad=False),
        torch.randn(batch_size, 64, 5, 3, device="cpu", requires_grad=False),
    ]
    network.skip_connections = dummy_skips

    def temp_forward(x):
        target_size = network.target_size
        batch_size = x.size(0)
        encoder_skips = network.skip_connections
        x = network.fc(x).view(batch_size, network.hidden_channels[0], network.pool, network.pool)
        skip3 = encoder_skips[2]
        skip3 = F.adaptive_avg_pool2d(skip3, (network.pool, network.pool))
        x = torch.cat([x, skip3], dim=1)
        x = network.deconv1(x)
        skip2 = encoder_skips[1]
        skip2 = F.adaptive_avg_pool2d(skip2, (4, 4))
        x = torch.cat([x, skip2], dim=1)
        x = network.deconv2(x)
        skip1 = encoder_skips[0]
        skip1 = F.adaptive_avg_pool2d(skip1, (8, 8))
        x = torch.cat([x, skip1], dim=1)
        x = network.deconv3(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        x = network.final_activation(x)
        return x

    network.forward = temp_forward
    dummy_input = torch.randn(*input_size, device="cpu", requires_grad=False)
    network.to("cpu").eval()
    torch.onnx.export(
        network,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}},
        verbose=False,
    )
    network.forward = original_forward
    network.skip_connections = original_skips
    network.to(original_device)
    if original_train_mode:
        network.train()
    print(f"Successfully exported map_decoder to {os.path.join(save_path)}")
    print(f"  - Input dims: {decoder_input_dims} (matches MapModule_info['output_channels']=32)")
    print(f"  - Skip channels: [16,32,64] (matches MapEncoder hidden_channels)")
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

   
    original_skips = model.skip_connections
  
    batch_size = input_size[0] 
    dummy_skips = [
        torch.randn(batch_size, 32, 16, 16, device='cpu', requires_grad=False),  # skip1: (B,32,16,16)
        torch.randn(batch_size, 64, 8, 8, device='cpu', requires_grad=False),   # skip2: (B,64,8,8)
        torch.randn(batch_size, 64, 4, 4, device='cpu', requires_grad=False)    # skip3: (B,64,4,4)
    ]
    model.skip_connections = dummy_skips

    dummy_input = torch.randn(*input_size, device='cpu', requires_grad=False)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "visual_token": {0: "batch_size"},  
            "recon_image": {0: "batch_size"}
        },
        verbose=False
    )


    model.skip_connections = original_skips  
    print(f"[INFO] Exported ImageDecoder to {save_path}")


import torch
import torch.nn.functional as F 
import os


def export_map_decoder_as_onnx(network, input_size, path, name, input_names, output_names):

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, name)

    first_param = next(network.parameters())
    original_device = first_param.device
    original_skips = network.skip_connections  
    original_train_mode = network.training
    original_forward = network.forward 

    decoder_input_dims = 32
    decoder_hidden_ch = network.hidden_channels  
    decoder_pool = network.pool 

    batch_size = input_size[0]
    dummy_skips = [
        torch.randn(batch_size, 16, 17, 11, device='cpu', requires_grad=False),  # skip1: 16通道（MapEncoder conv1）
        torch.randn(batch_size, 32, 9, 6, device='cpu', requires_grad=False),  # skip2: 32通道（MapEncoder conv2）
        torch.randn(batch_size, 64, 5, 3, device='cpu', requires_grad=False)  # skip3: 64通道（MapEncoder conv3）
    ]
    network.skip_connections = dummy_skips

    def temp_forward(x):
        target_size = network.target_size  
        batch_size = x.size(0)
        encoder_skips = network.skip_connections

        x = network.fc(x).view(batch_size, network.hidden_channels[0], network.pool, network.pool)

        skip3 = encoder_skips[2]
        skip3 = F.adaptive_avg_pool2d(skip3, (network.pool, network.pool))
        x = torch.cat([x, skip3], dim=1)
        x = network.deconv1(x)

        skip2 = encoder_skips[1]
        skip2 = F.adaptive_avg_pool2d(skip2, (4, 4))
        x = torch.cat([x, skip2], dim=1)
        x = network.deconv2(x)

        skip1 = encoder_skips[0]
        skip1 = F.adaptive_avg_pool2d(skip1, (8, 8))
        x = torch.cat([x, skip1], dim=1)
        x = network.deconv3(x)  

        x = F.interpolate(
            x,
            size=target_size,
            mode='bilinear',
            align_corners=True
        )

        x = network.final_activation(x)
        return x

    network.forward = temp_forward


    dummy_input = torch.randn(*input_size, device='cpu', requires_grad=False)
    network.to('cpu').eval()

    torch.onnx.export(
        network, 
        dummy_input,
        save_path,
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={  
            input_names[0]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"}
        },
        verbose=False
    )

    network.forward = original_forward 
    network.skip_connections = original_skips
    network.to(original_device)
    if original_train_mode:
        network.train()

