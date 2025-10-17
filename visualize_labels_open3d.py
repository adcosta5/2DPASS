#!/usr/bin/env python3
"""
Visualize SemanticKITTI-style .bin pointclouds with .label predictions using Open3D.

Usage examples:
  - Visualize a single pair:
      python scripts/visualize_labels_open3d.py --bin /path/to/000000.bin --label /path/to/000000.label

  - Visualize all predictions in a submit folder:
      python scripts/visualize_labels_open3d.py --submit_dir /path/to/submit_YYYY_MM_DD

  - Save colored PLY instead of opening a viewer:
      python scripts/visualize_labels_open3d.py --bin 000000.bin --label 000000.label --out 000000_colored.ply

The script reads the color_map from config/label_mapping/semantic-kitti.yaml by default.
"""

import argparse
import os
import struct
import numpy as np
import yaml
try:
    import open3d as o3d
except Exception:
    o3d = None


def read_bin(path):
    # SemanticKITTI velodyne bin: float32 x,y,z,intensity per point
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3], points[:, 3]


def read_label(path):
    # labels stored as uint32 per-point (SemanticKITTI format)
    labels = np.fromfile(path, dtype=np.uint32)
    # Some label files use lower 16 bits for label (legacy). Handle both.
    if labels.dtype == np.uint32:
        # mask out instance id (upper 16 bits) if present
        labels = labels & 0xFFFF
    return labels.astype(np.int32)


def load_colormap(yaml_path=None):
    # If user provided a yaml path, use it. Otherwise search common locations so the
    # script works whether placed in `scripts/` or moved to the repo root or executed
    # from another working directory.
    if yaml_path is not None:
        yaml_path = os.path.abspath(yaml_path)
        if not os.path.exists(yaml_path):
            raise RuntimeError(f'Provided yaml not found: {yaml_path}')
    else:
        candidates = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Likely places: next to script, in parent directory (if script moved), or cwd/config
        candidates.append(os.path.join(script_dir, 'config', 'label_mapping', 'semantic-kitti.yaml'))
        candidates.append(os.path.join(script_dir, '..', 'config', 'label_mapping', 'semantic-kitti.yaml'))
        cwd = os.path.abspath(os.getcwd())
        candidates.append(os.path.join(cwd, 'config', 'label_mapping', 'semantic-kitti.yaml'))

        # Walk up a few levels to find a repo root containing config/label_mapping
        def walk_up_for(base):
            p = base
            for _ in range(6):
                candidates.append(os.path.join(p, 'config', 'label_mapping', 'semantic-kitti.yaml'))
                parent = os.path.dirname(p)
                if parent == p:
                    break
                p = parent

        walk_up_for(script_dir)
        walk_up_for(cwd)

        yaml_path = None
        for c in candidates:
            if c and os.path.exists(os.path.abspath(c)):
                yaml_path = os.path.abspath(c)
                break
        if yaml_path is None:
            raise RuntimeError('Could not locate semantic-kitti.yaml; provide path with --yaml')

    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    color_map = cfg.get('color_map', None)
    if color_map is None:
        raise RuntimeError('No color_map found in yaml at %s' % yaml_path)
    # convert to RGB floats in [0,1]
    cmap = {}
    for k, v in color_map.items():
        cmap[int(k)] = np.array(v[::-1], dtype=np.float32) / 255.0  # yaml is BGR, reverse to RGB
    return cmap


def color_points_by_label(labels, cmap, default_color=(0.5, 0.5, 0.5)):
    rgb = np.zeros((labels.shape[0], 3), dtype=np.float32)
    for k, color in cmap.items():
        rgb[labels == k] = color
    # fallback
    mask = (rgb.sum(axis=1) == 0)
    if mask.any():
        rgb[mask] = np.array(default_color, dtype=np.float32)
    return rgb


def make_o3d_point_cloud(pts, colors=None):
    if o3d is None:
        raise RuntimeError('open3d is not installed. Please pip install open3d')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def key_based_player(frames, cmap, out_prefix=None):
    """Persistent window viewer using Open3D VisualizerWithKeyCallback.

    Key bindings:
      Right/ 'n' : next frame
      Left/ 'p'  : previous frame
      q / ESC    : quit
    """
    import open3d as o3d_local

    n = len(frames)
    if n == 0:
        print('No frames to show')
        return

    # small LRU cache for recent frames (to avoid re-reading rapidly)
    cache = {}
    cache_size = 10

    def load_frame(i):
        if i in cache:
            return cache[i]
        seq, frame_name, bin_path, label_path = frames[i]
        pts, intensity = read_bin(bin_path)
        labels = read_label(label_path)
        colors = color_points_by_label(labels, cmap)
        if len(cache) >= cache_size:
            # remove oldest
            cache.pop(next(iter(cache)))
        cache[i] = (pts, colors, seq, frame_name, label_path)
        return cache[i]

    idx = {'i': 0}

    vis = o3d_local.visualization.VisualizerWithKeyCallback()
    vis.create_window('2DPASS Viewer')

    # initial geometry
    pts, colors, seq0, frame0, label0 = load_frame(0)
    pcd = o3d_local.geometry.PointCloud()
    pcd.points = o3d_local.utility.Vector3dVector(pts)
    pcd.colors = o3d_local.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    def update_vis(i):
        pts, colors, seq, frame_name, label_path = load_frame(i)
        pcd.points = o3d_local.utility.Vector3dVector(pts)
        pcd.colors = o3d_local.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        print(f'Frame {i+1}/{n} — seq: {seq} — file: {frame_name}')
        if out_prefix:
            out_name = os.path.join(os.path.dirname(label_path), frame_name.replace('.bin', '_colored.ply'))
            o3d_local.io.write_point_cloud(out_name, pcd)
            print('Wrote', out_name)

    def next_cb(vis_obj):
        idx['i'] = min(n - 1, idx['i'] + 1)
        update_vis(idx['i'])
        return False

    def prev_cb(vis_obj):
        idx['i'] = max(0, idx['i'] - 1)
        update_vis(idx['i'])
        return False

    def quit_cb(vis_obj):
        vis_obj.destroy_window()
        return False

    # register multiple keys
    vis.register_key_callback(ord('N'), next_cb)
    vis.register_key_callback(ord('n'), next_cb)
    vis.register_key_callback(262, next_cb)  # right arrow (GLFW)
    vis.register_key_callback(ord('P'), prev_cb)
    vis.register_key_callback(ord('p'), prev_cb)
    vis.register_key_callback(263, prev_cb)  # left arrow (GLFW)
    vis.register_key_callback(ord('Q'), quit_cb)
    vis.register_key_callback(ord('q'), quit_cb)
    vis.register_key_callback(256, quit_cb)  # ESC

    # start
    update_vis(0)
    vis.run()
    vis.destroy_window()





def find_pairs_in_submit(submit_dir):
    # SemanticKITTI submit layout: submit_*/sequences/<seq>/predictions/*.label
    pairs = []
    seq_dir = os.path.join(submit_dir, 'sequences')
    if not os.path.exists(seq_dir):
        return pairs
    for seq in sorted(os.listdir(seq_dir)):
        pred_dir = os.path.join(seq_dir, seq, 'predictions')
        if not os.path.isdir(pred_dir):
            continue
        for fname in sorted(os.listdir(pred_dir)):
            if not fname.endswith('.label'):
                continue
            label_path = os.path.join(pred_dir, fname)
            # guess corresponding bin in original dataset location won't be here; user should supply --bin_root
            pairs.append((label_path, seq, fname))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin', help='Single .bin file (point cloud)')
    parser.add_argument('--label', help='Single .label file (predictions)')
    parser.add_argument('--submit_dir', help='Submit directory produced by --submit_to_server')
    parser.add_argument('--bin_root', help='Root folder where original velodyne .bin files live (used with --submit_dir)')
    parser.add_argument('--yaml', help='Path to semantic-kitti label mapping yaml', default=None)
    parser.add_argument('--out', help='If set, writes colored PLY instead of opening viewer')
    parser.add_argument('--point_color_intensity', action='store_true', help='Color points by grayscale intensity if no label')
    # (cache size option removed - GUI player was disabled)
    args = parser.parse_args()

    cmap = load_colormap(args.yaml)

    if args.label and args.bin:
        pts, intensity = read_bin(args.bin)
        labels = read_label(args.label)
        if labels.shape[0] != pts.shape[0]:
            print('Warning: label length %d != point count %d' % (labels.shape[0], pts.shape[0]))
        colors = color_points_by_label(labels, cmap)
        pcd = make_o3d_point_cloud(pts, colors)
        if args.out:
            o3d.io.write_point_cloud(args.out, pcd)
            print('Wrote', args.out)
        else:
            o3d.visualization.draw_geometries([pcd])
        return

    if args.submit_dir:
        pairs = find_pairs_in_submit(args.submit_dir)
        if len(pairs) == 0:
            print('No predictions found under', args.submit_dir)
            return
        # require bin_root to map back to velodyne files (assumes same structure sequences/<seq>/velodyne/xxxx.bin)
        if args.bin_root is None:
            print('When visualizing a submit directory, you should provide --bin_root where original .bin files live')
            print('Example: --bin_root /path/to/SemanticKITTI/dataset/sequences')
            return
        # Build an ordered list of frames: (seq, frame_name, bin_path, label_path)
        frames = []
        for label_path, seq, fname in pairs:
            frame_name = fname.replace('.label', '.bin')
            candidate_bin = os.path.join(args.bin_root, seq, 'velodyne', frame_name)
            if not os.path.exists(candidate_bin):
                print('Missing bin for', label_path, 'expected at', candidate_bin)
                continue
            frames.append((seq, frame_name, candidate_bin, label_path))

        if len(frames) == 0:
            print('No valid frame pairs found (missing corresponding .bin files)')
            return

        # Use the persistent key-based viewer that updates a single window
        key_based_player(frames, cmap, out_prefix=args.out)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
