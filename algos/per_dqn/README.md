# Prioritized Experience Replay

The standard DQN uses a buffer to break up the correlation between experiences and uniform random samples for each 
batch. Instead of just randomly sampling from the buffer prioritized experience replay (PER) prioritizes these samples
based on training loss. This concept was introduced in the paper 
[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

Essentially we want to train more on the samples that suprise the agent.

The priority of each sample is defined below where 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(i) = \frac{P^\alpha_i}{\sum_kP_k^\alpha}" />

where pi is the priority of the ith sample in the buffer and
𝛼 is the number that shows how much emphasis we give to the priority. If 𝛼 = 0 , our
sampling will become uniform as in the classic DQN method. Larger values for 𝛼 put
more stress on samples with higher priority

Its important that new samples are set to the highest priority so that they are sampled soon. This however introduces
bias to new samples in our dataset. In order to compensate for this bias, the value of the weight is defined as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;w_i = (N . P(i))^{-\beta}" />

Wher beta is a hyper parameter between 0-1. When beta is 1 the bias is fully compensated. However authors noted that 
in practice it is better to start beta with a small value near 0 and slowly increase it to 1.

### Benefits

- The benefits of this technique are that the agent sees more samples that it struggled with and gets more
chances to improve upon it.

### PER Memory Buffer

First step is to replace the standard experience replay buffer with the prioritized experience replay buffer. This
is pretty large (100+ lines) so I wont go through it here. There are two buffers implemented. The first is a naive
list based buffer found in memory.PERBuffer and the second is more efficient buffer using a Sum Tree datastructure. 

The list based version is simpler, but has a sample complexity of O(N). The Sum Tree in comparison has a complexity
of O(1) for sampling and O(logN) for updating priorities.

### Update loss function

The next thing we do is to use the sample weights that we get from PER. Add the following code to the end of the
loss function. This applies the weights of our sample to the batch loss. Then we return the mean loss and weighted loss
for each datum, with the addition of a small epsilon value.

````python

    # explicit MSE loss
    loss = (state_action_values - expected_state_action_values) ** 2

    # weighted MSE loss
    weighted_loss = batch_weights * loss

    # return the weighted_loss for the batch and the updated weighted loss for each datum in the batch
    return weighted_loss.mean(), (weighted_loss + 1e-5).data.cpu().numpy()

````

## Results

The results below improved stability and faster performance growth. 

### Pong

#### PER DQN
 
This seems to have improved the DQN the least. By the 500k mark it looks like the agent has achieved a pretty good
policy, but its a slower progression and a bit more noisy. 
 
![Noisy DQN Results](../../docs/images/pong_per_dqn_baseline_results.png)

#### DQN vs PER DQN 

In comparison to the base DQN, the PER DQN shows comparable performance. Infact the final averaged reward is 20 on 
the PER DQN and 21 on the DQN. Also, the loss of the PER DQN looks a little more volatile than the base DQN. One point
of interest is that the the episode steps peak and lower much faster than the base DQN, implying that the PER DQN agent
starts to beat the heuristic pong AI sooner than DQN. 

There are a few reasons why PER DQN may not be showing better improvements:

- PER introduces two new hyper parameters, so some tuning may need to be done for these.
- The implementation of the PER buffer is very simplistic and should be improved.


![Noisy DQN Comparison](../../docs/images/pong_per_dqn_dqn_comparison.png)

 - Orange: DQN

 - Pink: PER DQN