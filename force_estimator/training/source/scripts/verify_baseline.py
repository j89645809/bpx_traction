#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import yaml


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="."); ap.add_argument("--output",required=True); a=ap.parse_args()
    root=Path(a.root).resolve(); here=Path(__file__).parents[1]
    manifest=yaml.safe_load((here/"contracts/baseline_manifest.yaml").read_text())
    frozen=root/manifest["frozen_directory"]; checks={}
    for name,expected in manifest["assets"].items():
        path=frozen/name; actual=sha256(path) if path.is_file() else None
        checks[name]={"expected":expected,"actual":actual,"passed":actual==expected}
    deploy=yaml.safe_load((frozen/"deploy.yaml").read_text())
    dim=sum(len(v["scale"]) for v in deploy["observations"].values())
    checks["observation_dimension"]={"expected":73,"actual":dim,"passed":dim==73}
    checks["action_dimension"]={"expected":12,"actual":len(deploy["actions"]["JointPositionAction"]["scale"]),"passed":len(deploy["actions"]["JointPositionAction"]["scale"])==12}
    try:
        commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        status=subprocess.check_output(["git","-C",str(root),"status","--short"],text=True).splitlines()
    except Exception: commit=None; status=[]
    report={"passed":all(v["passed"] for v in checks.values()),"checks":checks,"git_commit":commit,"working_tree_changes":status}
    Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    if not report["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
