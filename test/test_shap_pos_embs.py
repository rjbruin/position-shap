import pytest
import subprocess

def test_errorless_run_shap_vit_ape():
    pos_emb = 'absolute_learnable'
    args = f"--debug --dryrun --shap_debug --exp_name=debug --log_first_batch --test --shap --seed=0 --shap_seed=0 --n_epochs=1 --check_val_every_n_epoch=10 --num_workers=2 --shuffle_val --adv_augmentations --random-erasing=1.0 --mixup-alpha=0.8 --cutmix-alpha=0.8 --dataset=cifar10 --internal_img_size=32 --opt=adam --batch_size=16 --val_batch_size=4 --model=own-vit --dataset_policy=own-vit --pos_emb={pos_emb} --net=vit --vit_config=cifar_ganiv2_dropout"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_errorless_run_shap_vit_rpe():
    pos_emb = 'relative_learnable'
    args = f"--debug --dryrun --shap_debug --exp_name=debug --log_first_batch --test --shap --seed=0 --shap_seed=0 --n_epochs=1 --check_val_every_n_epoch=10 --num_workers=2 --shuffle_val --adv_augmentations --random-erasing=1.0 --mixup-alpha=0.8 --cutmix-alpha=0.8 --dataset=cifar10 --internal_img_size=32 --opt=adam --batch_size=16 --val_batch_size=4 --model=own-vit --dataset_policy=own-vit --pos_emb={pos_emb} --net=vit --vit_config=cifar_ganiv2_dropout"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_errorless_run_shap_vit_fourier():
    pos_emb = 'fourier'
    args = f"--debug --dryrun --shap_debug --exp_name=debug --log_first_batch --test --shap --seed=0 --shap_seed=0 --n_epochs=1 --check_val_every_n_epoch=10 --num_workers=2 --shuffle_val --adv_augmentations --random-erasing=1.0 --mixup-alpha=0.8 --cutmix-alpha=0.8 --dataset=cifar10 --internal_img_size=32 --opt=adam --batch_size=16 --val_batch_size=4 --model=own-vit --dataset_policy=own-vit --pos_emb={pos_emb} --net=vit --vit_config=cifar_ganiv2_dropout"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_errorless_run_shap_vit_cape():
    pos_emb = 'cape'
    args = f"--debug --dryrun --shap_debug --exp_name=debug --log_first_batch --test --shap --seed=0 --shap_seed=0 --n_epochs=1 --check_val_every_n_epoch=10 --num_workers=2 --shuffle_val --adv_augmentations --random-erasing=1.0 --mixup-alpha=0.8 --cutmix-alpha=0.8 --dataset=cifar10 --internal_img_size=32 --opt=adam --batch_size=16 --val_batch_size=4 --model=own-vit --dataset_policy=own-vit --pos_emb={pos_emb} --net=vit --vit_config=cifar_ganiv2_dropout"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_errorless_run_shap_vit_rope():
    pos_emb = 'rope'
    args = f"--debug --dryrun --shap_debug --exp_name=debug --log_first_batch --test --shap --seed=0 --shap_seed=0 --n_epochs=1 --check_val_every_n_epoch=10 --num_workers=2 --shuffle_val --adv_augmentations --random-erasing=1.0 --mixup-alpha=0.8 --cutmix-alpha=0.8 --dataset=cifar10 --internal_img_size=32 --opt=adam --batch_size=16 --val_batch_size=4 --model=own-vit --dataset_policy=own-vit --pos_emb={pos_emb} --net=vit --vit_config=cifar_ganiv2_dropout"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"