#!/usr/bin/env python3
"""Fine-tune the primary TCN from multiple frozen NPZ sources."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from world_force.model import MissingStateTCN
from train_estimator import Normalized,task_loss


def load_many(paths):
    xs=[];ys=[]
    for path in paths:
        z=np.load(path);xs.append(z["sensor_history"]);ys.append(z["privileged_targets"])
    return torch.from_numpy(np.concatenate(xs)),torch.from_numpy(np.concatenate(ys))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--train",nargs="+",required=True);ap.add_argument("--validation",nargs="+",required=True);ap.add_argument("--initialize",required=True);ap.add_argument("--output",required=True);ap.add_argument("--epochs",type=int,default=8);ap.add_argument("--lr",type=float,default=2e-4);ap.add_argument("--batch-size",type=int,default=512);ap.add_argument("--velocity-weight",type=float,default=1);ap.add_argument("--height-weight",type=float,default=8);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    tx,ty=load_many(a.train);vx,vy=load_many(a.validation);init=torch.load(Path(a.initialize)/"estimator.pt",map_location="cpu");mean=init["mean"];std=init["std"];model=MissingStateTCN(input_dim=init["input_dim"],channels=init.get("channels",64));model.load_state_dict(init["model"]);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);loader=DataLoader(TensorDataset(tx,ty),batch_size=a.batch_size,shuffle=True,num_workers=0);best=float("inf");history=[]
    for epoch in range(a.epochs):
        model.train();total=0;t0=time.time()
        for x,y in loader:
            p,c=model(((x-mean)/std).to(device));loss=task_loss(p,y.to(device),c,a.velocity_weight,a.height_weight);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(x)
        model.eval();parts=[]
        with torch.no_grad():
            for start in range(0,len(vx),4096):
                p,c=model(((vx[start:start+4096]-mean)/std).to(device));parts.append(task_loss(p,vy[start:start+4096].to(device),c,a.velocity_weight,a.height_weight).item()*len(p))
        valid=sum(parts)/len(vx);row={"epoch":epoch+1,"train_loss":total/len(tx),"validation_loss":valid,"seconds":time.time()-t0};history.append(row);print(row,flush=True)
        if valid<best:best=valid;torch.save({"model":model.state_dict(),"mean":mean,"std":std,"input_dim":init["input_dim"],"channels":init.get("channels",64)},out/"estimator.pt")
    ck=torch.load(out/"estimator.pt",map_location="cpu");m=MissingStateTCN(input_dim=ck["input_dim"],channels=ck["channels"]).eval();m.load_state_dict(ck["model"]);wrap=Normalized(m,ck["mean"],ck["std"]).eval();example=torch.zeros(1,25,ck["input_dim"]);torch.jit.trace(wrap,example).save(str(out/"estimator.ts"));torch.onnx.export(wrap,example,out/"estimator.onnx",input_names=["sensor_history"],output_names=["values","confidence"],dynamic_axes={"sensor_history":{0:"batch"},"values":{0:"batch"},"confidence":{0:"batch"}},opset_version=17);np.savez(out/"normalization.npz",mean=mean.numpy(),std=std.numpy());(out/"training_report.json").write_text(json.dumps({"schema_version":"gate41-direction-primary-v16","initialized_from":a.initialize,"train_sources":a.train,"validation_sources":a.validation,"train_samples":len(tx),"validation_samples":len(vx),"best_validation_loss":best,"epochs":history},indent=2)+"\n")


if __name__=="__main__":main()
