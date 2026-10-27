#!/usr/bin/env python3
"""Write the explicit MuJoCo tether protocol atomically."""
from __future__ import annotations
import argparse, os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",required=True,choices=("legacy_body","world_fixed")); ap.add_argument("--force",nargs=3,type=float,required=True); ap.add_argument("--frame-id",required=True); ap.add_argument("--point-body",nargs=3,type=float,default=(0,0,0)); ap.add_argument("--output",default="/tmp/gate41_sim_force.txt"); a=ap.parse_args()
    if any(c.isspace() for c in a.frame_id): raise SystemExit("frame-id cannot contain whitespace")
    target=Path(a.output); tmp=target.with_suffix(target.suffix+".tmp")
    values=[a.mode,*a.force,a.frame_id,*a.point_body]
    tmp.write_text(" ".join(map(str,values))+"\n"); os.replace(tmp,target)


if __name__=="__main__": main()
