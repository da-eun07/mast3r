#!/usr/bin/env python3
# demo_local.py
# Copyright (C) 2024-present Naver Corporation.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Local script referencing glomap-based pipeline
# --------------------------------------------------------

import os
import argparse
import tempfile
import shutil
import copy
import torch
import PIL.Image
import numpy as np

# ========== MASt3R imports =============
from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.mapping import (kapture_import_image_folder_or_list,
                                   run_mast3r_matching,
                                   glomap_run_mapper)
from mast3r.image_pairs import make_pairs

# ========== kapture-based COLMAP bridging ===========
from kapture.converter.colmap.database_extra import kapture_to_colmap
from kapture.converter.colmap.database import COLMAPDatabase

# ========== Additional imports ==========
import pycolmap
from dust3r.utils.image import load_images  # for loading/resizing images
# If "dust3r" is not installed, replace with your own load_images function or another approach.

class GlomapRecon:
    def __init__(self, world_to_cam, intrinsics, points3d, imgs):
        self.world_to_cam = world_to_cam
        self.intrinsics = intrinsics
        self.points3d = points3d
        self.imgs = imgs


class GlomapReconState:
    def __init__(self, glomap_recon, should_delete=False, cache_dir=None, outfile_name=None):
        self.glomap_recon = glomap_recon
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def main():
    parser = argparse.ArgumentParser("Local glomap-based pipeline with MASt3R.")
    parser.add_argument("--input_dir", required=True,
                        help="Folder with images.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to store colmap.db and the final reconstruction.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to local .pth checkpoint or HF name for MASt3R.")
    parser.add_argument("--glomap_bin", type=str, default=None,
                        help="Path to glomap binary. If None, we use standard colmap incremental_mapper.")
    parser.add_argument("--device", default="cuda",
                        help="Compute device (e.g. 'cuda', 'cpu').")
    parser.add_argument("--size", type=int, default=512,
                        help="Max dimension for 'load_images(...)'.")
    parser.add_argument("--skip_verification", action="store_true",
                        help="If set, skip pycolmap.verify_matches step.")
    parser.add_argument("--scene_graph", default="complete",
                        help="Which scene graph mode for 'make_pairs'. E.g. 'complete', 'retrieval-20-10', etc.")
    parser.add_argument("--retrieval_model", default=None,
                        help="Optional retrieval checkpoint path for retrieval-based scene graph.")
    parser.add_argument("--conf_thr", type=float, default=1.001,
                        help="Confidence threshold in 'run_mast3r_matching'.")
    parser.add_argument("--pixel_tol", type=int, default=5,
                        help="Pixel tolerance in 'run_mast3r_matching'.")
    parser.add_argument("--min_len_track", type=int, default=3,
                        help="Min track length in final colmap matches.")
    parser.add_argument("--shared_intrinsics", action='store_true',
                        help="If set, pass 'use_single_camera=True' to kapture_import_image_folder_or_list.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    colmap_dp_path = os.path.join(args.output_dir, "colmap.db")

    # 1) Load the model
    print(f"Loading MASt3R model from {args.weights}")
    model = AsymmetricMASt3R.from_pretrained(args.weights).to(args.device)

    # 2) Collect images from input_dir, in ascending order
    #    We'll pass them to 'load_images(...)' for resizing
    #    Then we build a list of file paths
    exts = (".png", ".jpg", ".jpeg")
    filelist = []
    for fn in sorted(os.listdir(args.input_dir)):
        if any(fn.lower().endswith(e) for e in exts):
            filelist.append(os.path.join(args.input_dir, fn))
    if not filelist:
        print(f"No valid images in {args.input_dir}")
        return
    print(f"[INFO] Found {len(filelist)} images. Loading them with size={args.size}...")

    # We'll just load them to confirm the shape (optional).
    # 'run_mast3r_matching' itself calls inference for each pair. So let's also store "images" for building pairs.
    images_d = load_images(filelist, size=args.size, verbose=True)
    for i, d in enumerate(images_d):
        d["idx"] = i  # ensure an index

    # 3) Build the scene graph pairs
    #    If "retrieval" in args.scene_graph, we also build sim_matrix with a retrieval model
    sim_matrix = None
    if "retrieval" in args.scene_graph:
        # We must have a retrieval model
        if not args.retrieval_model:
            print("Error: scene_graph has 'retrieval', but no --retrieval_model given.")
            return
        print(f"Loading retrieval model from {args.retrieval_model}")
        from mast3r.retrieval.processor import Retriever
        retr = Retriever(args.retrieval_model, backbone=model, device=args.device)
        with torch.no_grad():
            sim_matrix = retr(filelist)
        del retr
    pairs = make_pairs(images_d, scene_graph=args.scene_graph, symmetrize=False, sim_mat=sim_matrix)
    print(f"[INFO] Built {len(pairs)} pairs with scene_graph={args.scene_graph}")
    
    # 4) Make a minimal kapture dataset from the image folder
    root_path = os.path.commonpath(filelist)
    filelist_relpath = [
        os.path.relpath(filename, root_path).replace("\\", "/")
        for filename in filelist
    ]
    from mast3r.colmap.mapping import kapture_import_image_folder_or_list
    kdata = kapture_import_image_folder_or_list((root_path, filelist_relpath),
                                                use_single_camera=args.shared_intrinsics)
    image_pairs = [
        (filelist_relpath[img1['idx']], filelist_relpath[img2['idx']])
        for img1, img2 in pairs
    ]

    # 5) Create a colmap db
    if os.path.isfile(colmap_dp_path):
        os.remove(colmap_dp_path)
    os.makedirs(os.path.dirname(colmap_dp_path), exist_ok=True)

    colmap_db = COLMAPDatabase.connect(colmap_dp_path)
    # Insert cameras/images with kapture => colmap
    try:
        kapture_to_colmap(kdata, root_path, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        ######### TODO: TypeError: unhashable type: 'dict' #########
        colmap_image_pairs = run_mast3r_matching(model, args.size, 16, args.device,
                                                 kdata, root_path, image_pairs, colmap_db,
                                                 dense_matching=False, pixel_tol=args.pixel_tol,
                                                 conf_thr=args.conf_thr, skip_geometric_verification=False,
                                                 min_len_track=args.min_len_track)
        colmap_db.close()
    except Exception as e:
        print(f'Error {e}')
        colmap_db.close()
        return
    
    if len(colmap_image_pairs) == 0:
        print("No matches found. Exiting.")
        return
    
    # colmap db is now full, run colmap
    colmap_world_to_cam = {}
    print("verify_matches")
    f = open(args.output_dir + '/pairs.txt', "w")
    for image_path1, image_path2 in colmap_image_pairs:
        f.write("{} {}\n".format(image_path1, image_path2))
    f.close()
    pycolmap.verify_matches(colmap_dp_path, args.output_dir + '/pairs.txt')

    reconsturction_path = os.path.join(args.output_dir, "reconstruction")
    if os.path.isdir(reconsturction_path):
        shutil.rmtree(reconsturction_path)
    os.makedirs(reconsturction_path, exist_ok=True)
    glomap_run_mapper(args.glomap_bin, colmap_dp_path, reconsturction_path, root_path)

    output_recon = pycolmap.Reconstruction(os.path.join(reconsturction_path, "0"))
    print("[INFO] Reconstruction summary:")
    print(output_recon.summary())

    colmap_world_to_cam = {}
    colmap_intrinsic = {}
    colmap_image_id_to_name = {}
    images = {}
    num_reg_images = output_recon.num_reg_images()
    for idx, (colmap_imgid, colmap_image) in enumerate(output_recon.images.items()):
        colmap_image_id_to_name[colmap_imgid] = colmap_image.name
        if callable(colmap_image.cam_from_world.matrix):
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix()
        else:
            colmap_world_to_cam[colmap_imgid] = colmap_image.cam_from_world.matrix
        camera = output_recon.cameras[colmap_image.camera_id]
        K = np.eye(3)
        K[0, 0] = camera.focal_length_x
        K[1, 1] = camera.focal_length_y
        K[0, 2] = camera.principal_point_x
        K[1, 2] = camera.principal_point_y
        colmap_intrinsic[colmap_imgid] = K

        with PIL.Image.open(os.path.join(root_path, colmap_image.name)) as img:
            images[colmap_imgid] = np.asarray(img)

        if idx + 1 == num_reg_images:
            break

    # 6) Write the final points3D.txt
    points3D = []
    num_points3D = output_recon.num_points3D()
    for idx, (pt3d_id, pts3d) in enumerate(output_recon.points3D.items()):
        points3D.append((pts3d.xyz, pts3d.color))
        if idx + 1 == num_points3D:
            break

    # optionally export a .ply
    out_ply = os.path.join(reconsturction_path, "points3D_colmap.ply")
    output_recon.export_PLY(out_ply)
    print(f"[DONE] wrote final points3D to {out_ply}")

if __name__ == "__main__":
    main()
