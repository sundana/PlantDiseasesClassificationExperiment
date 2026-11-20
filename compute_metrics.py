"""
Compute accuracy and macro-averaged precision, recall and F1 (Table "tab:prf").

Usage A: from saved predictions (one CSV per model and dataset)
    python compute_metrics.py preds --csv preds_V3L_SE4_plantvillage.csv
    python compute_metrics.py preds --csv preds_V3L_SE4_plantdoc.csv --mapping mapping.json
  CSV columns: true,pred   (class names; for PlantDoc, `true` = PlantDoc class,
                            `pred` = predicted PlantVillage class)
  mapping.json: {"Tomato Early blight leaf": "Tomato___Early_blight", ...}  (20 entries)

Usage B: run a checkpoint on an ImageFolder directory and write the CSV
    python compute_metrics.py infer --ckpt model.pt --data test_dir --out preds.csv \
        --classes classes_plantvillage.txt
  (the checkpoint must be a full torch model saved with torch.save(model, path);
   classes file lists the 29 PlantVillage class names in output-index order)

Definitions follow Section "Evaluation metrics": per-class P, R, F1 from the
confusion matrix, macro-averaged over the evaluated classes (29 for PlantVillage,
20 matched classes for PlantDoc). Classes never predicted get precision 0.
Predictions of unmapped PlantVillage classes count as false negatives only.
"""
import argparse, csv, json
import numpy as np


def macro_metrics(y_true, y_pred, classes):
    """y_true / y_pred: lists of class names; classes: evaluated classes (K)."""
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    tp = np.zeros(K); fp = np.zeros(K); fn = np.zeros(K)
    correct = 0
    for t, p in zip(y_true, y_pred):
        ti = idx[t]
        if p == t:
            tp[ti] += 1; correct += 1
        else:
            fn[ti] += 1
            if p in idx:            # unmapped predictions are not a FP of any evaluated class
                fp[idx[p]] += 1
    prec = np.divide(tp, tp + fp, out=np.zeros(K), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(K), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    acc = correct / len(y_true)
    return {"accuracy": 100 * acc, "precision": 100 * prec.mean(), "recall": 100 * rec.mean(),
            "f1": 100 * f1.mean(),
            "per_class": {c: (100 * prec[i], 100 * rec[i], 100 * f1[i], int(tp[i] + fn[i]))
                          for i, c in enumerate(classes)}}


def cmd_preds(a):
    rows = list(csv.DictReader(open(a.csv)))
    y_true = [r["true"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    if a.mapping:
        m = json.load(open(a.mapping))
        keep = [(m[t], p) for t, p in zip(y_true, y_pred) if t in m]   # only matched classes
        y_true = [t for t, _ in keep]; y_pred = [p for _, p in keep]
        classes = sorted(set(m.values()))
    else:
        classes = sorted(set(y_true))
    r = macro_metrics(y_true, y_pred, classes)
    print(f"N = {len(y_true)}, K = {len(classes)}")
    print(f"Accuracy {r['accuracy']:.2f} | Precision {r['precision']:.2f} | "
          f"Recall {r['recall']:.2f} | F1 {r['f1']:.2f}")
    print(f"LaTeX cells: {r['accuracy']:.2f} & {r['precision']:.2f} & {r['recall']:.2f} & {r['f1']:.2f}")
    if a.per_class:
        for c, (p, rc, f, n) in r["per_class"].items():
            print(f"  {c:45s} P={p:6.2f} R={rc:6.2f} F1={f:6.2f} n={n}")


def cmd_infer(a):
    import torch
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = datasets.ImageFolder(a.data, transform=tf)
    out_classes = [l.strip() for l in open(a.classes) if l.strip()]
    model = torch.load(a.ckpt, map_location="cpu", weights_only=False).eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true", "pred"])
        with torch.no_grad():
            for x, y in dl:
                p = model(x).argmax(1)
                for yi, pi in zip(y.tolist(), p.tolist()):
                    w.writerow([ds.classes[yi], out_classes[pi]])
    print("written", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("preds"); p1.add_argument("--csv", required=True)
    p1.add_argument("--mapping"); p1.add_argument("--per-class", action="store_true")
    p2 = sub.add_parser("infer"); p2.add_argument("--ckpt", required=True)
    p2.add_argument("--data", required=True); p2.add_argument("--out", required=True)
    p2.add_argument("--classes", required=True)
    a = ap.parse_args()
    cmd_preds(a) if a.cmd == "preds" else cmd_infer(a)
