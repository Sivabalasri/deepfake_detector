import json
import pandas as pd
import os

def save_epoch_log(log_data):
    csv_path = "logs/experiment_history.csv"

    df = pd.DataFrame([log_data])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

def save_final_metrics(metrics):
    with open("logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)