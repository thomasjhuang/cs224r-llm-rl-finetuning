#!/usr/bin/env python3
"""
Monitor DPO training metrics and alert when evaluation might be needed
"""
import wandb
import time
import argparse

def monitor_training(project_name, run_name=None):
    """Monitor wandb metrics and suggest evaluation points"""
    
    api = wandb.Api()
    
    if run_name:
        run = api.run(f"{project_name}/{run_name}")
    else:
        # Get latest run
        runs = api.runs(project_name)
        if not runs:
            print("No runs found!")
            return
        run = runs[0]
    
    print(f"Monitoring run: {run.name}")
    
    last_step = 0
    eval_triggered = False
    
    while True:
        try:
            history = run.scan_history()
            metrics = list(history)
            
            if not metrics:
                time.sleep(10)
                continue
                
            latest = metrics[-1]
            current_step = latest.get('global_step', 0)
            
            if current_step > last_step:
                # Check for concerning patterns
                reward_acc = latest.get('rewards_train/accuracies', 0)
                loss = latest.get('loss', float('inf'))
                
                print(f"Step {current_step}: Reward Acc={reward_acc:.3f}, Loss={loss:.3f}")
                
                # Trigger evaluation suggestions
                if current_step >= 500 and not eval_triggered:
                    if reward_acc < 0.52:
                        print("🚨 WARNING: Low reward accuracy! Consider stopping for evaluation.")
                    elif reward_acc > 0.6:
                        print("✅ GOOD: High reward accuracy! Safe to continue or evaluate.")
                    eval_triggered = True
                
                last_step = current_step
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("Monitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="qwen2-dpo-conservative")
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    
    monitor_training(args.project, args.run) 