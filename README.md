# TextureGAN

Pytorch implementation of TextureGAN

```bash
python main.py --display_port 7770 --load -1 --load_D -1 --gpu 0 --model scribbler --feature_weight 10 --pixel_weight_ab 500 --pixel_weight_l 100 --style_weight 1 --discriminator_weight 0 --learning_rate 1e-2 --learning_rate_D 1e-4 --load_dir /home/varun/projects/TextureGAN_pytorch/save_dir --save_dir /home/varun/projects/TextureGAN_pytorch/save_dir --data_path /home/varun/datasets/texturegan/training_handbags_pretrain --batch_size 45 --save_every 500 --num_epoch 100000
```

## Using Docker

```
nvidia-docker run --name=varun_texturegan_box  -it -v /home/vagrawal38:/home/vagrawal38:rw --net=host -p 8770:8770 -p 8771:8771 -p 8772:8772 -p 8773:8773 -p 80:80 janesjanes/myenv:current /bin/bash
```
