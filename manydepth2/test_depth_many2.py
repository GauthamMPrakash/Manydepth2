import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from .utils import readlines
from .options import MonodepthOptions
from manydepth2 import datasets, networks
from .datasets import VideoDataset
from .layers import transformation_from_parameters, disp_to_depth
import tqdm
import torch.nn.functional as F
cv2.setNumThreads(0)
splits_dir = "splits"
STEREO_SCALE_FACTOR = 5.4
import matplotlib.pyplot as plt
from core_gm.gmflow.gmflow.gmflow import GMFlow

import os
ADDRESS = "run/code/1_mgdepth/MANYDEPTH2/logs/mono_img_mg/"
# The Address where you want to save generated predicted result.

if not os.path.exists(ADDRESS):
    os.mkdir(ADDRESS)
if not os.path.exists(ADDRESS + '/mono_img_mg'):
    os.mkdir(ADDRESS + '/mono_img_mg')

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def evaluate(opt):
    """Evaluate depth estimation on a video file with real-time visualization"""
    
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    frames_to_load = [0, -1]  # Current and previous frame
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if model path is provided
    if not opt.load_weights_folder:
        print("Error: --load_weights_folder is required")
        return
    
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        f"Cannot find a folder at {opt.load_weights_folder}"
    
    print(f"-> Loading weights from {opt.load_weights_folder}")
    print(f"-> Processing video: {opt.video_path}")
    
    # Load model weights
    coarse_encoder_path = os.path.join(opt.load_weights_folder, "mono_encoder.pth")
    coarse_decoder_path = os.path.join(opt.load_weights_folder, "mono_depth.pth")
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    
    # Load encoder dictionary to get dimensions
    encoder_dict = torch.load(encoder_path, weights_only=True, map_location=device)
    
    try:
        HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
    except KeyError:
        HEIGHT, WIDTH = opt.height, opt.width
    
    print(f"-> Using image size: {HEIGHT}x{WIDTH}")
    
    # Create VideoDataset
    dataset = VideoDataset(
        video_path=opt.video_path,
        height=HEIGHT,
        width=WIDTH,
        frame_idxs=frames_to_load,
        num_scales=4,
        is_train=False
    )
    
    print(f"-> Video has {len(dataset)} frames")
    
    # Setup models
    # ------------ Loading Flow Model ------------
    feature_channels = 128
    num_scales = 1
    upsample_factor = 8
    num_head = 1
    attention_type = 'swin'
    ffn_dim_expansion = 4
    num_transformer_layers = 6
    
    attn_splits_list = [2]
    corr_radius_list = [-1]
    prop_radius_list = [-1]
    
    model_gmflow = GMFlow(
        feature_channels=feature_channels,
        num_scales=num_scales,
        upsample_factor=upsample_factor,
        num_head=num_head,
        attention_type=attention_type,
        ffn_dim_expansion=ffn_dim_expansion,
        num_transformer_layers=num_transformer_layers
    ).to(device)
    
    gmflow_checkpoint = torch.load('pretrained/gmflow_sintel-0c07dcb3.pth', weights_only=True, map_location=device)
    weights = gmflow_checkpoint['model'] if 'model' in gmflow_checkpoint else gmflow_checkpoint
    model_gmflow.load_state_dict(weights)
    model_gmflow.eval()
    
    print(" ------------ Loading Pose Network ------------ ")
    pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"), weights_only=True, map_location=device)
    pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"), weights_only=True, map_location=device)
    
    pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
    
    pose_enc.load_state_dict(pose_enc_dict, strict=True)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)
    pose_enc.eval()
    pose_dec.eval()
    
    if torch.cuda.is_available():
        pose_enc.cuda()
        pose_dec.cuda()
    
    print(" ------------ Loading Coarse Encoder ------------ ")
    coarse_encoder_dict = torch.load(coarse_encoder_path, weights_only=True, map_location=device)
    coarse_encoder = networks.hrnet18(pretrained=False)
    coarse_encoder.load_state_dict({k: v for k, v in coarse_encoder_dict.items() if k in coarse_encoder.state_dict()})
    coarse_encoder.num_ch_enc = [64, 18, 36, 72, 144]
    
    coarse_depth_decoder = networks.HRDepthDecoder(coarse_encoder.num_ch_enc)
    coarse_depth_decoder.load_state_dict(torch.load(coarse_decoder_path, weights_only=True, map_location=device))
    
    coarse_encoder.eval()
    coarse_depth_decoder.eval()
    
    if torch.cuda.is_available():
        coarse_encoder.cuda()
        coarse_depth_decoder.cuda()
    
    print(" ------------ Loading Fine Encoder ------------ ")
    encoder_opts = dict(
        num_layers=opt.num_layers,
        pretrained=False,
        input_width=encoder_dict['width'],
        input_height=encoder_dict['height'],
        adaptive_bins=True,
        min_depth_bin=0.1,
        max_depth_bin=20.0,
        depth_binning=opt.depth_binning,
        num_depth_bins=opt.num_depth_bins
    )
    
    encoder = networks.multihrnet18_flow00(**encoder_opts)
    encoder.num_ch_enc = [64, 18, 36, 72, 144]
    
    depth_decoder = networks.HRDepthDecoder(encoder.num_ch_enc)
    
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    depth_decoder.load_state_dict(torch.load(decoder_path, weights_only=True, map_location=device))
    
    encoder.eval()
    depth_decoder.eval()
    
    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()
    
    min_depth_bin = encoder_dict.get('min_depth_bin', 0.1)
    max_depth_bin = encoder_dict.get('max_depth_bin', 20.0)
    
    # Setup visualization windows
    if opt.display_window:
        cv2.namedWindow('Input Frame', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Input Frame', 640, 192)
        cv2.resizeWindow('Depth Map', 640, 192)
        print("-> Press 'q' to quit, 'p' to pause/resume, 's' to save current frame")
    
    # Ensure output directory exists
    if opt.save_frames or opt.display_window:
        os.makedirs(opt.output_dir, exist_ok=True)
    
    # Process video frames
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=2, pin_memory=True)
    
    paused = False
    frame_count = 0
    save_next_frame = False
    
    print("-> Starting video processing...")
    
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            print("frame", i)

            if opt.display_window:
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("-> Quitting...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"-> {'Paused' if paused else 'Resumed'}")
                elif key == ord('s'):
                    save_next_frame = True
                    print(f"-> Saving frame {i}")
                
                if paused:
                    continue
            
            # Skip first frame since we need previous frame for flow
            if i == 0:
                continue
            
            # Prepare pose computation
            pose_feats = {f_i: data["color", f_i, 0] for f_i in frames_to_load}
            if torch.cuda.is_available():
                pose_feats = {k: v.cuda() for k, v in pose_feats.items()}

            # Compute pose from current to previous frame
            pose_inputs = [pose_feats[-1], pose_feats[0]]
            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
            axisangle, translation = pose_dec(pose_inputs)
            pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)
            
            # Prepare data for depth estimation
            lookup_frames = data[('color', -1, 0)].unsqueeze(1)
            relative_poses = pose.unsqueeze(1)
            K = data[('K', 2)]
            invK = data[('inv_K', 2)]
            
            if torch.cuda.is_available():
                lookup_frames = lookup_frames.cuda()
                relative_poses = relative_poses.cuda()
                K = K.cuda()
                invK = invK.cuda()
                data["color", 0, 0] = data["color", 0, 0].cuda()
                data["color", -1, 0] = data["color", -1, 0].cuda()

            # Compute optical flow
            results_dict = model_gmflow(
                data["color", 0, 0] * 255.,
                data["color", -1, 0] * 255.,
                attn_splits_list=attn_splits_list,
                corr_radius_list=corr_radius_list,
                prop_radius_list=prop_radius_list,
            )
            
            flow_preds = results_dict['flow_preds'][-1]
            flow_preds = F.interpolate(flow_preds, scale_factor=0.25, mode='bilinear', align_corners=False)
            flow_preds /= 4.
            
            # Compute coarse depth
            input_color = data[('color', 0, 0)]
            min_depth, max_depth = 0.1, 100.
            feats = coarse_encoder(input_color)
            depth_dict = coarse_depth_decoder(feats)
            disp = depth_dict[("disp", 0)]
            _, depth = disp_to_depth(disp, min_depth, max_depth)
            coarse_depth = F.interpolate(depth, [HEIGHT//4, WIDTH//4], mode="bilinear", align_corners=False)
            
            # Final depth estimation
            output, lowest_cost, costvol = encoder(
                input_color, lookup_frames, relative_poses,
                flow_preds, coarse_depth, K, invK,
                min_depth_bin, max_depth_bin, True
            )
            
            output = depth_decoder(output)
            pred_disp, pred_depth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            
            # Convert to numpy for visualization
            input_np = input_color[0].permute(1, 2, 0).cpu().numpy()
            input_np = (input_np * 255).astype(np.uint8)
            input_np = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
            
            depth_np = pred_depth[0, 0].cpu().numpy()
            depth_colored = cv2.applyColorMap(
                (depth_np / depth_np.max() * 255).astype(np.uint8),
                cv2.COLORMAP_MAGMA
            )
            
            # Display frames
            if opt.display_window:
                cv2.imshow('Input Frame', input_np)
                cv2.imshow('Depth Map', depth_colored)
            
            # Save frames
            if opt.save_frames or save_next_frame:
                save_path_rgb = os.path.join(opt.output_dir, f'frame_{i:06d}_rgb.png')
                save_path_depth = os.path.join(opt.output_dir, f'frame_{i:06d}_depth.png')
                cv2.imwrite(save_path_rgb, input_np)
                cv2.imwrite(save_path_depth, depth_colored)
                if save_next_frame:
                    save_next_frame = False  # Reset save flag
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"-> Processed {frame_count} frames")
    
    if opt.display_window:
        cv2.destroyAllWindows()
    
    print(f"-> Finished processing {frame_count} frames")
    if opt.save_frames:
        print(f"-> Frames saved to: {opt.output_dir}")

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())


