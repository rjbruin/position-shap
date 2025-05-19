import wandb

api = wandb.Api()
run = api.run("tudcv/vit_position_embeddings/3bszgqpa")
print(run.lastHistoryStep)
print(run.scan_history(keys=['val/acc'], min_step=run.lastHistoryStep))
run.summary.update()