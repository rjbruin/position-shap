import pytest
import subprocess

def test_errorless_run_toy_sequential_shap():
    args = "--debug --exp_name=testcase_0 --no-training --test --shap --seed=0 --check_val_every_n_epoch=10 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_errorless_run_toy_shap_spatial_features():
    args = "--debug --exp_name=testcase_1 --shap_spatial_features --no-training --test --shap --seed=0 --check_val_every_n_epoch=10 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_errorless_run_toy_shap_image_channels_features():
    args = "--debug --exp_name=testcase_2 --shap_image_channels_features --no-training --test --shap --seed=0 --check_val_every_n_epoch=10 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_errorless_run_toy_shap_single_batch_bg():
    args = "--debug --exp_name=testcase_3 --shap_single_batch_bg --no-training --test --shap --seed=0 --check_val_every_n_epoch=10 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_errorless_run_toy_shap_fold_size():
    args = "--debug --exp_name=testcase_4 --shap_fold_size=4 --no-training --test --shap --seed=0 --check_val_every_n_epoch=10 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_errorless_run_toy_batched_shap():
    args = "--debug --exp_name=testcase_5 --shap_start_at_batch 0 --shap_stop_at_batch 1 --no-training --test --shap --seed=0 --check_val_every_n_epoch=10 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    args = "--debug --exp_name=testcase_5 --shap_start_at_batch 1 --no-training --test --shap --seed=0 --check_val_every_n_epoch=10 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

if __name__ == "__main__":
    pytest.main()