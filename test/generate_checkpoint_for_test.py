import subprocess

def generate_checkpoint_for_test():
    args = "--debug --exp_name=checkpoint-for-test-24 --save_checkpoint --seed=0 --check_val_every_n_epoch=100 --num_workers=2 --dataset=toy --toy_size=8 --internal_img_size=8 --opt=adam --batch_size=16 --model=own-vit --dataset_policy=own-vit --pos_emb=absolute_learnable --net=toy --patch_size=2 --toy_n_blocks=1 --toy_pos_init=trunc_normal:0.02 --toy_mlp_d=8 --toy_pooling=avg --toy_pos_add=concat_equald"
    command = ['python', 'train.py'] + args.split()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

    # Print the path to the checkpoint
    print(result.stdout)

if __name__ == "__main__":
    generate_checkpoint_for_test()