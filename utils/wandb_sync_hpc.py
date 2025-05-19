"""Sync W&B runs based on a given Slurm ID."""
import sys
import os
import tqdm


if __name__ == '__main__':
    slurm_ids = list(map(int, sys.argv[1:]))
    print(f"Received slurm IDs {slurm_ids}")

    print(f"Ready to sync:")

    to_sync = []
    for slurm_id in slurm_ids:
        filename = f"/home/nfs/robertjanbruin/bulk-home/out/run-{slurm_id}.out"

        if not os.path.exists(filename):
            print(f"Run {slurm_id} hasn't started yet.")
            continue

        finished_training = False
        finished_syncing = False
        with open(filename, 'r') as f:
            try:
                for line in f:
                    if "wandb: Run data is saved locally in " in line:
                        line = line.strip()
                        words = line.split("wandb: Run data is saved locally in ")
                        local_dir = words[1]
                    elif "wandb: Syncing run " in line:
                        line = line.strip()
                        words = line.split("wandb: Syncing run ")
                        run_name = words[1]
                    elif "wandb: 🚀 View run at " in line:
                        line = line.strip()
                        words = line.split("wandb: 🚀 View run at ")
                        run_link = words[1]
                    elif "wandb: Waiting for W&B process to finish... (success)." in line:
                        finished_syncing = True
                    elif "`Trainer.fit` stopped" in line:
                        finished_training = True
            except UnicodeDecodeError:
                print(f"Could not read {slurm_id}: file {filename}")
                continue

        if finished_training and finished_syncing:
            print(f"  {run_link}  {run_name}")
            to_sync.append((slurm_id, local_dir, run_name, run_link))
        else:
            print(f"  {slurm_id} not finished yet; skipping.")

    answer = input("Continue? [y/n] ")
    if answer.lower() in ["y","yes"]:
        pass
    elif answer.lower() in ["n","no"]:
        print("Exiting.")
        exit(0)
    else:
        print("Unrecognized answer. Exiting.")
        exit(0)

    for slurm_id, local_dir, run_name, run_link in tqdm.tqdm(to_sync):
        os.system(f"wandb sync {local_dir}")
        os.system(f"scancel {slurm_id}")