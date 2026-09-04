import csv
import os

runs_dir = 'runs/train'
for run_name in os.listdir(runs_dir):
    csv_path = os.path.join(runs_dir, run_name, 'results.csv')
    if not os.path.exists(csv_path):
        continue

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(cleaned)

    if not rows:
        continue

    map50_95_key = 'metrics/mAP50-95(B)'
    map50_key = 'metrics/mAP50(B)'
    prec_key = 'metrics/precision(B)'
    recall_key = 'metrics/recall(B)'
    epoch_key = 'epoch'

    best_row = max(rows, key=lambda r: float(r[map50_95_key]))
    last_row = rows[-1]

    print(f'========== {run_name} ==========')
    print(f'  Total epochs: {len(rows)}')
    print(f'  Best epoch:   {int(float(best_row[epoch_key]))}')
    print(f'  Best mAP50:      {float(best_row[map50_key]):.4f}')
    print(f'  Best mAP50-95:   {float(best_row[map50_95_key]):.4f}')
    print(f'  Best Precision:  {float(best_row[prec_key]):.4f}')
    print(f'  Best Recall:     {float(best_row[recall_key]):.4f}')
    print()
