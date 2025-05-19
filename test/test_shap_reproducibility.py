import pytest
import subprocess
import math

def _run_shap(args, get_values=True, train_path='train.py'):
    command = ['python', train_path] + args.split()
    result1 = subprocess.run(command, capture_output=True, text=True)
    assert result1.returncode == 0, f"Run 1 failed with error: {result1.stderr}"

    if not get_values:
        return {}

    output = result1.stdout
    # Find the index of the line that says "Validate metric"
    lines = output.split("\n")
    validate_metric_start = None
    validate_metric_end = None
    break_lines = 0
    for i, line in enumerate(lines):
        if "Validate metric" in line:
            validate_metric_start = i
            continue
        if validate_metric_start is not None and line.startswith("─────"):
            break_lines += 1
            if break_lines == 2:
                validate_metric_end = i
    assert validate_metric_start is not None, "Could not find 'Validate metric' line in output: " + output
    assert validate_metric_end is not None, "Could not find line that demarcates end of validation metrics in output: " + output

    # Save all listed W&B logged values
    wandb_logged_values = {}
    for line in lines[validate_metric_start+2:validate_metric_end]:
        assert len(line.split()) == 2, "Could not parse line" + line + "in lines: " + "\n".join(lines[validate_metric_start+2:-2])
        metric_name = line.split()[0].strip()
        metric_value = float(line.split()[1].strip())
        wandb_logged_values[metric_name] = metric_value

    # See if the script saved the checkpoint
    checkpoint_path = None
    for line in lines:
        if line.startswith("Checkpoint saved to"):
            checkpoint_path = line.split()[-1]
            break

    return wandb_logged_values, checkpoint_path

def _test_values_close(vals1, vals2):
    # Compare the logged values
    for key in vals1:
        if key.startswith('p-shap/p-values') or key.startswith('p-shap/values'):
            assert key in vals2, f"Key {key} not found in second run"
            assert math.isclose(vals1[key],
                                vals2[key]), \
                                f"Values for key {key} are not equal: {vals1[key]} - {vals2[key]} = {vals1[key] - vals2[key]}"

def test_shap_identical_seeds():
    args = "--debug --test --shap --exp_name=testcase_seed0 --resume_from_checkpoint=test/checkpoint_for_shap_reproducibility.ckpt --seed=0 --check_val_every_n_epoch=100 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg --toy_pos_add=concat_equald"
    values_run1, _ = _run_shap(args)
    values_run2, _ = _run_shap(args)
    _test_values_close(values_run1, values_run2)

def test_shap_saved_and_resumed():
    args = "--debug  --test --shap --seed=0 --check_val_every_n_epoch=10 --n_epochs 100 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg --toy_pos_add=concat_equald"
    args_save = "--exp_name=testcase_save --save_checkpoint"
    args_resume = lambda path: f"--exp_name=testcase_resume --resume_from_checkpoint={path}"

    initial_values, ckpt_path = _run_shap(args + " " + args_save)
    resumed_values, _ = _run_shap(args + " " + args_resume(ckpt_path))
    _test_values_close(initial_values, resumed_values)


# def test_shap_sequential_vs_batched():
#     args = "--debug --exp_name=testcase_seq --test --shap --seed=0 --check_val_every_n_epoch=10 --n_epochs 100 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
#     values_sequential = _run_shap(args)

#     args = "--debug --exp_name=testcase_batched --shap_stop_at_batch=1 --test --shap --seed=0 --check_val_every_n_epoch=10 --n_epochs 100 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
#     _ = _run_shap(args, get_values=False)
#     args = "--debug --exp_name=testcase_batched --shap_start_at_batch=1 --test --shap --seed=0 --check_val_every_n_epoch=10 --n_epochs 100 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
#     values_batched = _run_shap(args)

#     _test_values_close(values_sequential, values_batched)

if __name__ == "__main__":
    pytest.main()