To train a new model, you'll need to edit the file nn.py
1. Set the variable plot_dir to the path of the location where the results need to be stored. The training checkpoints can be found at plot_dir/checkpoints.
2. Set signal_path to the location of .mat file of injected signals.
3. Set noise_path to the location of .mat file of background noise.
4. If you'd like to change which cWB parameters are used as input, change the variable cols_used accordingly

To use an existing model to run predictions on new data, you'll need to edit reuse_model.py
1. Set the variable data1 to the path of the unlabelled input data
2. Set cols_used to reflect the cWB parameters used in training
3. The python script needs to be run along with the location of the checkpoint file as a command line argument. For instance, if you'd like to use a checkpoint named model5 located at resuls/rho_g8_no_chirp/model5.ckpt.meta you'll have to use the following call `python code/reuse_model.py results/rho_g8_no_chirp/model5.ckpt`
