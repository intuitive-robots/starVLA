#!/usr/bin/env python3
import argparse
import glob
import json
import pathlib
from PIL import Image
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from scripts.collect_libero_object_trace_overlays import overlay, parse_trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-label", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10103)
    a = p.parse_args()
    inp, out = pathlib.Path(a.input_dir), pathlib.Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows=[]; examples=[]
    for meta_path in sorted(glob.glob(str(inp / "task_*_ep_*.json"))):
        meta=json.load(open(meta_path)); stem=pathlib.Path(meta_path).stem
        agent=np.asarray(Image.open(inp/f"{stem}_agent.png")); wrist=np.asarray(Image.open(inp/f"{stem}_wrist.png"))
        rows.append((meta,agent)); examples.append({"image":[agent,wrist],"lang":meta["description"]})
    client=WebsocketClientPolicy(a.host,a.port)
    response=client.predict_action({"examples":examples,"unnorm_key":None,"do_sample":False,"use_ddim":True,"num_ddim_steps":10})
    client.close(); texts=response["data"].get("cot_text")
    if not isinstance(texts,(list,tuple)) or len(texts)!=len(rows):
        raise RuntimeError(f"Expected {len(rows)} traces, got {texts!r}")
    result=[]
    for (meta,image),text in zip(rows,texts):
        points=parse_trace(text); task_id=meta["task_id"]; episode=meta["episode_idx"]
        overlay(image,points,f"{a.model_label} | task {task_id} | {len(points)} points").save(out/f"task_{task_id:02d}_ep_{episode:02d}.png")
        result.append({**meta,"model":a.model_label,"cot_text":text,"trace_2d":points})
    json.dump(result,open(out/"traces.json","w"),indent=2)
    print(f"{a.model_label}: {sum(len(x['trace_2d'])==5 for x in result)}/{len(result)} valid five-point traces")


if __name__ == "__main__": main()
