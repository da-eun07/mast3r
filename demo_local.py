#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import argparse
import numpy as np
import torch

# Import MASt3R
from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.utils.misc import mkdir_for, hash_md5

from dust3r.utils.image import load_images
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

"""
demo_local.py: A local command-line script that:
  1) Reads images from a directory using dust3r's load_images()
  2) Runs the MASt3R "sparse_global_alignment"
  3) Exports COLMAP-like text files (cameras.txt, images.txt, points3D.txt)
  4) Writes a PLY of the final 3D points

Usage Example:
  python demo_local.py \
      --input_dir data/composite/images \
      --output_dir output \
      --weights checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
      --lr1 0.07 --niter1 300 --lr2 0.01 --niter2 300
"""

def create_colmap_text_output(sparse_ga, image_list, output_dir):
    """
    Creates colmap-like text files:
      output_dir/sparse/0/cameras.txt
      output_dir/sparse/0/images.txt
      output_dir/sparse/0/points3D.txt
    Also writes a PLY (points3D.ply).
    """
    os.makedirs(os.path.join(output_dir, "sparse/0"), exist_ok=True)

    # 1) Read final data from MASt3R's "SparseGA" object
    cam2w = sparse_ga.get_im_poses()  # shape [N,4,4]
    focals = sparse_ga.get_focals()   # shape [N]
    ppoints = sparse_ga.get_principal_points()  # shape [N,2]
    # we can also get the original image dimension from image_list
    # get 3D points: we choose "sparse" anchors
    pts3d_list = sparse_ga.get_sparse_pts3d()
    colors_list = sparse_ga.get_pts3d_colors()

    # Merge all sparse pts3d
    all_points = []
    point_id = 0
    for i, p3d_i in enumerate(pts3d_list):
        col_i = colors_list[i]
        p3d_i = p3d_i.cpu().numpy()
        for j in range(p3d_i.shape[0]):
            xyz = p3d_i[j]
            ccc = col_i[j]*255.0
            all_points.append((point_id, xyz, ccc))
            point_id += 1

    # 2) Write cameras.txt (one camera per image => "PINHOLE")
    out_cam_path = os.path.join(output_dir, "sparse/0/cameras.txt")
    with open(out_cam_path, "w") as f:
        f.write("# Cameras\n")
        for i, item in enumerate(image_list):
            camera_id = i + 1
            # item["true_shape"] is (H, W)
            H, W = item["true_shape"][0]
            fx = float(focals[i].cpu().item())
            cx = float(ppoints[i][0].cpu().item()) * W
            cy = float(ppoints[i][1].cpu().item()) * H
            # For simplicity, pinhole => 4 params: fx,fy,cx,cy
            # We'll set fy=fx
            f.write(f"{camera_id} PINHOLE {W} {H} {fx} {fx} {cx} {cy}\n")

    # 3) Write images.txt
    from scipy.spatial.transform import Rotation as R
    out_im_path = os.path.join(output_dir, "sparse/0/images.txt")
    with open(out_im_path, "w") as f:
        f.write("# Images\n# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        for i, item in enumerate(image_list):
            image_id = i+1
            camera_id = i+1
            fname = os.path.basename(item["instance"])

            pose_cam2w = cam2w[i].cpu().numpy()
            w2cam = np.linalg.inv(pose_cam2w)
            Rmat = w2cam[:3, :3]
            tvec = w2cam[:3, 3]
            rot = R.from_matrix(Rmat)
            quat = rot.as_quat()  # (x,y,z,w)
            # colmap wants [qw,qx,qy,qz]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            tx, ty, tz = tvec
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {fname}\n")

    # 4) Write points3D.txt
    out_pts_path = os.path.join(output_dir, "sparse/0/points3D.txt")
    with open(out_pts_path, "w") as f:
        f.write("# 3D Points\n")
        for pid, (pidx, xyz_col, ccc) in enumerate(all_points):
            x, y, z = xyz_col
            r, g, b = ccc
            # track_length = 0, error=1.0
            f.write(f"{pidx} {x} {y} {z} {int(r)} {int(g)} {int(b)} 1.0 0\n")

    # 5) Write a PLY with the final pointcloud
    out_ply_path = os.path.join(output_dir, "sparse/0/points3D.ply")
    with open(out_ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex ")
        f.write(f"{len(all_points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pid, (pidx, xyz_col, ccc) in enumerate(all_points):
            x, y, z = xyz_col
            r, g, b = ccc
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
    print(f"[DONE] Wrote cameras, images, and points3D into '{output_dir}/sparse/0'.")


def main():
    parser = argparse.ArgumentParser("MASt3R local demonstration script (no Gradio).")
    parser.add_argument("--input_dir", required=True,
                        help="Path to directory containing images")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write final COLMAP-style reconstruction")
    parser.add_argument("--model_name", default=None, type=str,
                        help="A known HuggingFace or naver/<model> name, e.g. 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'")
    parser.add_argument("--weights", default=None, type=str,
                        help="Path to local .pth checkpoint for the model (overrides model_name).")
    parser.add_argument("--optim_level", default="refine+depth", help="coarse, refine, refine+depth", type=str)
    
    parser.add_argument("--lr1", default=0.07, type=float,
                        help="Learning rate (coarse alignment)")
    parser.add_argument("--niter1", default=300, type=int,
                        help="Number of iterations for coarse alignment")
    parser.add_argument("--lr2", default=0.01, type=float,
                        help="Learning rate (fine alignment)")
    parser.add_argument("--niter2", default=300, type=int,
                        help="Number of iterations for fine alignment")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device for inference, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--shared_intrinsics", action='store_true',
                        help="Optimize a single set of intrinsics for all images.")
    parser.add_argument("--matching_conf_thr", default=5.0, type=float,
                        help="Confidence threshold for fallback to DUSt3R regr if below. (Tune if needed).")

    # The dust3r/utils/image.load_images "size" param typically is 512 or so.
    # We'll default to 512 but you can override.
    parser.add_argument("--load_size", default=512, type=int,
                        help="Max dimension for loading images via load_images()")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load the model
    if args.weights is not None:
        model_ckpt = args.weights
        print(f"Loading local checkpoint = {model_ckpt}")
        mast3r_model = AsymmetricMASt3R.from_pretrained(model_ckpt).to(args.device)
    else:
        model_ckpt = args.model_name
        print(f"Loading huggingface model = {model_ckpt}")
        mast3r_model = AsymmetricMASt3R.from_pretrained(model_ckpt).to(args.device)

    # 2) Use dust3r's load_images(...) with auto-resize
    
    # load_images can handle the resizing automatically, ensuring multiples of 16, etc.
    images = load_images(args.input_dir, size=args.load_size, verbose=True)

    if len(images) < 2:
        print("[ERROR] Need at least 2 images. Found only one or zero.")
        return
    print(f"Loaded {len(images)} images from '{args.input_dir}' with auto-resizing to ~{args.load_size} px.")

    # 3) Make pairwise graph ("complete" for demonstration)
    
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True, sim_mat=None)
    print(f"Prepared {len(pairs)} pairs among {len(images)} images (complete graph).")

    # 4) Run the MASt3R sparse global alignment
    niter1 = args.niter1
    niter2 = args.niter2
    if args.optim_level == "coarse":
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)

    cache_dir = os.path.join(args.output_dir, "tmp_cache")
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(
        [img["instance"] for img in images], pairs, cache_dir,
        mast3r_model, device=args.device,
        lr1=args.lr1, niter1=niter1,
        lr2=args.lr2, niter2=niter2,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr
    )
    print("[INFO] sparse_global_alignment done.")

    # 5) Export the result to colmap style
    create_colmap_text_output(scene, images, args.output_dir)
    print(f"[DONE] Output saved in '{args.output_dir}'.")
    print("You can visualize 'points3D.ply' in meshlab, or check the text files in sparse/0/")

if __name__ == "__main__":
    main()
