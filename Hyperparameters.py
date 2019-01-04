BATCH_SIZE = 128        # minibatch size
BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise theta
OU_SIGMA = 0.01         # Ornstein-Uhlenbeck noise sigma


