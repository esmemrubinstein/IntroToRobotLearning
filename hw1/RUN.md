### Run Instructions

In Section 1.2, to get the expert results for the five environments, the following parameters in were used the runtime arguments block on the ipynb file in Google Colab:

```
ep_len = 1000 
num_agent_train_steps_per_iter = 2500 
n_iter = 1
batch_size = 1000 
eval_batch_size = 5000 
train_batch_size = 100 
max_replay_buffer_size = 1000000
n_layers = 5 
size = 64 
learning_rate = 4e-3
video_log_freq = -1 #@param 
scalar_log_freq = 1 #@param 
no_gpu = False #@param 
which_gpu = 0 #@param 
seed = 1 
```

In Section 1.3, keep the same parameters as above except for the `num_agent_train_steps_per_iter`.  In the `run_hw1_erubinst.ipynb` after the given train block, there is another block starting with the comment "code to run bc train multiple times with different parameter".  This block can be run to produce the results for changing a particular parameter (it is set to the `num_agent_train_steps_per_iter` that was used for my submission). The results will go to the same folder with each rollout in its own folder named after the rollout setting.  The next block in the file creates the plot based on the data file path specified in the variable `path`.  

In Section 2.2, use the following parameters (all not listed match above):

```
n_iter = 10
n_layers: 3
learning_rate: 5e-3 
```
After generated the outputs for 2 environments, there is another block in the ipynb file with the comment "Code to create dagger vs expert vs bc plot for ant and another environment" that can be used to create the plot for the DAgger vs BC vs Expert.






