"""Evaluate an exported BPX force estimator on an independent NPZ split."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='directory containing test.npz')
    ap.add_argument('--model', required=True, help='bpx_force.onnx')
    ap.add_argument('--normalization', required=True)
    ap.add_argument('--output', required=True)
    a=ap.parse_args()
    import onnxruntime as ort
    d=np.load(Path(a.dataset)/'test.npz')
    x=d['sensor_history'].astype(np.float32); y=d['force_world'].astype(np.float32)
    z=np.load(a.normalization); mean=z['mean'].astype(np.float32); std=z['std'].astype(np.float32)
    xn=((x-mean)/std).astype(np.float32)
    sess=ort.InferenceSession(a.model, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
    outs=sess.run(None, {sess.get_inputs()[0].name:xn})
    pred=outs[0]; conf=outs[1] if len(outs)>1 else np.ones((len(pred),1),np.float32)
    err=pred-y; mag=np.linalg.norm(y,axis=1); pmag=np.linalg.norm(pred,axis=1)
    mask=mag>=2.0
    cos=np.sum(pred[mask]*y[mask],axis=1)/(np.maximum(pmag[mask]*mag[mask],1e-6)); ang=np.degrees(np.arccos(np.clip(cos,-1,1)))
    result={'samples':int(len(y)), 'component_mae_N':np.mean(np.abs(err),axis=0).tolist(), 'component_rmse_N':np.sqrt(np.mean(err**2,axis=0)).tolist(), 'force_magnitude_mae_N':float(np.mean(np.abs(pmag-mag))), 'force_magnitude_rmse_N':float(np.sqrt(np.mean((pmag-mag)**2))), 'direction_mean_deg_above_2N':float(np.mean(ang)) if len(ang) else None, 'direction_p95_deg_above_2N':float(np.percentile(ang,95)) if len(ang) else None, 'confidence_mean':float(np.mean(conf)), 'confidence_mae':float(np.mean(np.abs(conf-(mag>2).astype(np.float32)[:,None])))}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
