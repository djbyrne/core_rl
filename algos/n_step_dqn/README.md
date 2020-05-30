# N Step DQN

N Step DQN was introduced in [Learning to Predict by the Methods
of Temporal Differences 
](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf). This method improves upon the original DQN by updating 
our Q values with the expected reward from multiple steps in the
future as opposed to the expected reward from the immediate next state. When getting the Q values for a state action 
pair using a single step which looks like this

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s_t,a_t)=r_t+{\gamma}\max_aQ(s_t+1,a_t+1)"/>

but because the Q function is recursive we can continue to roll this out into multiple steps, looking at the expected
return for each step into the future. 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s_t,a_t)=r_t+{\gamma}r_{t+1}+{\gamma}^2\max_{a'}Q(s_{t+2},a')"/>

The above example shows a 2-Step look ahead, but this could be rolled out to the end of the episode, which is just 
Monte Carlo learning. Although we could just do a monte carlo update and look forward to the end of the episode, it 
wouldn't be a good idea. Every time we take another step into the future, we are basing our approximation off our 
current policy. For a large portion of training, our policy is going to be less than optimal. For example, at the start
of training, our policy will be in a state of high exploration, and will be little better than random. 

---
**NOTE**

For each rollout step you must scale the discount factor accordingly by the number of steps. As you can see from the 
equation above, the second gamma value is to the power of 2. If we rolled this out one step further, we would use 
gamma to the power of 3 and so.

---

So if we are aproximating future rewards off a bad policy, chances are those approximations are going to be pretty 
bad and every time we unroll our update equation, the worse it will get. The fact that we are using an off policy method
like DQN with a large replay buffer will make this even worse, as there is a high chance that we will be training on 
experiences using an old policy that was worse than our current policy.

So we need to strike a balance between looking far enough ahead to improve the convergence of our agent, but not so far 
that are updates become unstable. In general, small values of 2-4 work best.  

### Benefits

- Multi-Step learning is capable of learning faster than typical 1 step learning methods.
- Note that this method introduces a new hyperparameter n. Although n=4 is generally a good starting point and provides
good results across the board.

### Implementation

#### Multi Step Buffer

`````python

    #  add this to the dqn network
    conv_out_size = self._get_conv_out(input_shape)

    # advantage head
    self.fc_adv = nn.Sequential(
        nn.Linear(conv_out_size, 256),
        nn.ReLU(),
        nn.Linear(256, n_actions)
    )

    # value head
    self.fc_val = nn.Sequential(
        nn.Linear(conv_out_size, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
``````

### Update Forward 
````python
    
    def forward(self, x):
        adv, val = self.adv_val(x)

        # return the full Q value which is value + adv while we pull the mean to 0
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256 # normalize
        conv_out = self.conv(fx).view(fx.size()[0], -1) 
        return self.fc_adv(conv_out), self.fc_val(conv_out)
````

## Results

The results below a noticeable improvement from the original DQN network. 

### Pong

#### Dueling DQN

Similar to the results of the DQN baseline, the agent has a period where the number of steps per episodes increase as 
it begins to 
hold its own against the heuristic oppoent, but then the steps per episode quickly begins to drop as it gets better 
and starts to 
beat its opponent faster and faster. There is a noticable point at step ~250k where the agent goes from losing to
winning.

As you can see by the total rewards, the dueling network's training progression is very stable and continues to trend 
upward until it finally plateus. 

![Dueling DQN Results](../../docs/images/pong_dueling_dqn_results.png)

#### DQN vs Dueling DQN 

In comparison to the base DQN, we see that the Dueling network's training is much more stable and is able to reach a
score in the high teens faster than the DQN agent. Even though the Dueling network is more stable and out performs DQN
early in training, by the end of training the two networks end up at the same point.

This could very well be due to the simplicity of the Pong environment. 

 - Orange: DQN

 - Red: Dueling DQN

![Dueling DQN Results](../../docs/images/pong_dueling_dqn_comparison.png)

https://arxiv.org/pdf/1901.07510.pdf