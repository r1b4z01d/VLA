"""Convert a LeRobot dataset's ABSOLUTE end-effector action into a DELTA (relative-to-current-pose)
action, to break the 'copycat' shortcut (the policy copying the current pose instead of using vision;
see scripts/debug_eval.py). Mirrors the leesweqq/ur5_vla_pick_cube_project action space.

For each frame:  action[:pose_dims] -= observation.state[:pose_dims]   (the commanded MOTION);
the trailing grasp/gripper dims stay ABSOLUTE. Images + state are left untouched — only the `action`
column (schema preserved) and its normalization stats in meta/stats.json change.

DEPLOY must add the delta back:  target_pose = current_ee_pose + action[:pose_dims]  (see eval_hw
--delta-action). Offline debug_eval needs no change (predicted delta vs recorded delta).

    .venv/bin/python scripts/make_delta_dataset.py --src outputs/datasets/pickup --dst outputs/datasets/pickup_delta
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--pose-dims", type=int, default=6,
                    help="leading action dims that are EE pose (delta'd vs state); rest stay absolute")
    args = ap.parse_args()
    n = args.pose_dims

    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        raise SystemExit("src and dst must differ")
    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    print(f"copying {args.src} -> {args.dst} …", flush=True)
    shutil.copytree(args.src, args.dst)

    parquets = sorted(glob.glob(os.path.join(args.dst, "data", "**", "*.parquet"), recursive=True))
    if not parquets:
        raise SystemExit("no data parquets found under dst/data")

    all_actions = []
    for pf in parquets:
        t = pq.read_table(pf)
        A = np.asarray(t["action"].to_pylist(), dtype=np.float32)              # [rows, action_dim]
        S = np.asarray(t["observation.state"].to_pylist(), dtype=np.float32)   # [rows, state_dim]
        A[:, :n] = A[:, :n] - S[:, :n]                                          # pose -> delta
        flat = pa.array(A.reshape(-1), type=pa.float32())
        new_col = pa.FixedSizeListArray.from_arrays(flat, A.shape[1])          # preserve schema
        idx = t.schema.get_field_index("action")
        t = t.set_column(idx, t.schema.field(idx), new_col)
        pq.write_table(t, pf)
        all_actions.append(A)

    allA = np.concatenate(all_actions)  # [total, action_dim]

    sp = os.path.join(args.dst, "meta", "stats.json")
    with open(sp) as f:
        stats = json.load(f)
    a = stats["action"]
    a["mean"] = allA.mean(0).tolist()
    a["std"] = np.maximum(allA.std(0), 1e-6).tolist()  # guard near-zero std -> no /0 in normalization
    a["min"] = allA.min(0).tolist()
    a["max"] = allA.max(0).tolist()
    for q in ("q01", "q10", "q50", "q90", "q99"):
        if q in a:
            a[q] = np.quantile(allA, int(q[1:]) / 100.0, axis=0).tolist()
    stats["action"] = a
    with open(sp, "w") as f:
        json.dump(stats, f, indent=2)

    ip = os.path.join(args.dst, "meta", "info.json")
    with open(ip) as f:
        info = json.load(f)
    if info.get("repo_id"):
        info["repo_id"] = "local/" + os.path.basename(os.path.normpath(args.dst))
        with open(ip, "w") as f:
            json.dump(info, f, indent=2)

    print(f"done: {len(allA)} frames -> delta actions ({n} pose dims). "
          f"action mean[:3]={[round(x, 4) for x in a['mean'][:3]]} (~0 = delta), "
          f"std[:3]={[round(x, 4) for x in a['std'][:3]]}", flush=True)


if __name__ == "__main__":
    main()
