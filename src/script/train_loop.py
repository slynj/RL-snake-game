import time
from src.model.train import train

def run_training_loop():
    while True:
        print("\n=== Starting New Training Session ===\n", flush=True)

        try:
            train()
        except Exception as e:
            print(f"Training encountered an error: {e}", flush=True)

        print("\n=== Training Complete. Restarting... ===\n", flush=True)

        time.sleep(10)

if __name__ == "__main__":
    run_training_loop()


'''
nohup python -m src.script.train_loop > log/train_log.txt 2>&1 &
pkill -f train_loop
'''