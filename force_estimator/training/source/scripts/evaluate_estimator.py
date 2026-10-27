#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import onnxruntime as ort


def metrics(y,p,c,times):
    active=np.linalg.norm(y[:,12:15],axis=1)>2
    cos=np.sum(y[active,12:15]*p[active,12:15],1)/(np.linalg.norm(y[active,12:15],axis=1)*np.linalg.norm(p[active,12:15],axis=1)+1e-9)
    return {
      "base_velocity_rmse_mps":float(np.sqrt(np.mean((p[:,:3]-y[:,:3])**2))),
      "base_height_rmse_m":float(np.sqrt(np.mean((p[:,3]-y[:,3])**2))),
      "contact_accuracy":float(np.mean((p[:,4:8]>=.5)==(y[:,4:8]>=.5))),
      "tether_force_rmse_n":float(np.sqrt(np.mean((p[:,12:15]-y[:,12:15])**2))),
      "tether_direction_mean_deg":float(np.degrees(np.arccos(np.clip(cos,-1,1))).mean()),
      "confidence_mean":float(c.mean()),
      "inference_ms_p50":float(np.percentile(times,50)),"inference_ms_p95":float(np.percentile(times,95)),"inference_ms_p99":float(np.percentile(times,99)),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--model",required=True); ap.add_argument("--output",required=True); a=ap.parse_args()
    sess=ort.InferenceSession(a.model,providers=["CPUExecutionProvider"]); result={}
    for split in ("test","ood"):
      d=np.load(Path(a.dataset)/f"{split}.npz"); x=d["sensor_history"]; y=d["privileged_targets"]
      pred=[]; conf=[]; times=[]
      for b in np.array_split(x,max(1,len(x)//128)):
        t=time.perf_counter(); p,c=sess.run(None,{"sensor_history":b}); times.append((time.perf_counter()-t)*1000/len(b)); pred.append(p); conf.append(c)
      result[split]=metrics(y,np.concatenate(pred),np.concatenate(conf),times)
    result["onnx_runtime"]="CPUExecutionProvider"; Path(a.output).write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__": main()
