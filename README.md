# README

The reconstructed images from CIFAR-10 are in the folder called "output_file"

# Train

For training, run the following program:

```python
python main.py
```

Parameters:

--**device**: str; The device you want to run the program. default=cpu

--**trainPath**: str; The folder of your train data.

--**valPath**: str; The folder of your validation/test data.

--**n_blocks**: int; The number of residual blocks, default=5

--**n_epochs**: int; epochs for training.

--**batch_size**: int; batch size for both train dataset and test dataset, default=64

--**num_workers**: int; default=0

--**lr**: float; learning rate, default=1e-3

## Test & reconstruct

```
python test.py --reconstruct
```

Parameters:

--**device**: str; The device you want to run the program. default=cpu

--**reconstruct**: int; If 1, the denoise images and raw images will be saved at --save_dir, default=0

--**n_blocks**: int; The number of residual blocks, default=5, should be the same as training.

--**batch_size**: int; batch size for test dataset, default=64

--**num_workers**: int; default=0

--**save_dir**: str; The folder to save the reconstructed images, default='./output_file/'
