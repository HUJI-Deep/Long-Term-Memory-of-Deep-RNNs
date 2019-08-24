# Long-Term-Memory-of-Deep-RNNs
This repository holds the source code for all experiments presented in:
https://arxiv.org/abs/1710.09431 - On the Long-Term Memory of Deep Recurrent Networks

# Required python packages
- tensorflow
- numpy
- sklearn
- argparse

# Comments
- The code of scoRNN.py is taken from  https://github.com/SpartinStuff/scoRNN. We strongly recommend you to read
  their paper - Scaled Cayley Orthogonal Recurrent Network (scoRNN) Cell - https://arxiv.org/abs/1707.09520.

# Command line Examples
Our results can be reproduced by running each experiment k times with the k appropriate configurations.
Command line examples are presented below - one configuration of each experiment.

## Copying Memory Task
cd copying_memory_task
python run_copying_task.py -num_iters 50000 -rnn_depth 2 -rnn_hidden_dim 64 -B 10

## Start-End Similarity Task
cd start_end_similarity
python run_SES.py -num_iters 50000 -rnn_depth 2 -rnn_hidden_dim 64 -T 80

## Permuted pixel-by-pixel MNIST
cd sequential_MNIST
python run_seq_MNIST.py -permute 1 -num_iters 50000 -rnn_depth 2 -rnn_hidden_dim 64
