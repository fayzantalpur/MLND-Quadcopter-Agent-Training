# Function to do the Agent Training
# Libraries Needed
import numpy as np
import pandas as pd
import random,copy,os, errno,csv
from physics_sim import PhysicsSim
from collections import namedtuple, deque
from keras import layers, models, optimizers, regularizers
from keras import backend as K


### Main Function To pass respective values
def QuadCopter_Train(Num_Episodes = 50,
                     Target_Pos = np.array([0., 0., 10.]),
                     File_Output = 'data.txt', 
                     Runtime = 5,
                     Init_Pose = np.array([0., 0., 0., 0., 0., 0.]),
                     Init_Velocities = np.array([0., 0., 0.]),
                     Init_Angle_Velocities = np.array([0., 0., 0.]),
                     Action_Repeat = 3,
                     State_Size = 6,
                     Action_Low = 0,
                     Action_High = 900,
                     Action_Size = 4,
                     Reward = '1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()',
                     Buffer_Size = 100000,
                     Batch_Size = 64,
                     Exploration_Mu = 0.0,
                     Exploration_Theta = 0.15,
                     Exploration_Sigma = 0.2,
                     DDPG_Gamma = 0.99,
                     DDPG_Tau = 0.01):
                     

#### 1.1 Task
    class Task():
        """Task (environment) that defines the goal and provides feedback to the agent."""
        def __init__(self, init_pose = Init_Pose, init_velocities = Init_Velocities, 
            init_angle_velocities = Init_Angle_Velocities, runtime = Runtime, target_pos = Target_Pos):
            """Initialize a Task object.
            Params
            ======
                init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
                init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
                init_angle_velocities: initial radians/second for each of the three Euler angles
                runtime: time limit for each episode
                target_pos: target/goal (x,y,z) position for the agent
            """
            # Simulation
            self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
            self.action_repeat = Action_Repeat

            self.state_size = self.action_repeat * State_Size
            self.action_low = Action_Low
            self.action_high = Action_High
            self.action_size = Action_Size

            # Goal
            self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

        def get_reward(self):
            """Uses current pose of sim to return reward."""
            reward = eval(Reward)
            return reward

        def step(self, rotor_speeds):
            """Uses action to obtain next state, reward, done."""
            reward = 0
            pose_all = []
            for _ in range(self.action_repeat):
                done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
                reward += self.get_reward() 
                pose_all.append(self.sim.pose)

            
            next_state = np.concatenate(pose_all)
            return next_state, reward, done

        def reset(self):
            """Reset the sim to start a new episode."""
            self.sim.reset()
            state = np.concatenate([self.sim.pose] * self.action_repeat)
            return state

#### 1.2 DDPG Agent
    class ReplayBuffer:
        """Fixed-size buffer to store experience tuples."""

        def __init__(self, buffer_size, batch_size):
            """Initialize a ReplayBuffer object.
            Params
            ======
                buffer_size: maximum size of buffer
                batch_size: size of each training batch
            """
            self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
            self.batch_size = batch_size
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        def add(self, state, action, reward, next_state, done):
            """Add a new experience to memory."""
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

        def sample(self, batch_size=64):
            """Randomly sample a batch of experiences from memory."""
            return random.sample(self.memory, k=self.batch_size)

        def __len__(self):
            """Return the current size of internal memory."""
            return len(self.memory)

    class Actor:
        """Actor (Policy) Model."""

        def __init__(self, state_size, action_size, action_low, action_high):
            """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                action_low (array): Min value of each action dimension
                action_high (array): Max value of each action dimension
            """
            self.state_size = state_size
            self.action_size = action_size
            self.action_low = action_low
            self.action_high = action_high
            self.action_range = self.action_high - self.action_low

            ######## Initialize any other variables here

            self.build_model()

        def build_model(self):
            """Build an actor (policy) network that maps states -> actions."""
            # Define input layer (states)
            states = layers.Input(shape=(self.state_size,), name='states')

            # Add hidden layers
            net = layers.Dense(units=32, activation='relu')(states)
            net = layers.Dense(units=64, activation='relu')(net)
            net = layers.Dense(units=32, activation='relu')(net)

            ########## Try different layer sizes, activations, add batch normalization, regularizers, etc.
       

            # Add final output layer with sigmoid activation
            raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
                name='raw_actions')(net)

            # Scale [0, 1] output for each action dimension to proper range
            actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                name='actions')(raw_actions)

            # Create Keras model
            self.model = models.Model(inputs=states, outputs=actions)

            # Define loss function using action value (Q value) gradients
            action_gradients = layers.Input(shape=(self.action_size,))
            loss = K.mean(-action_gradients * actions)

            ######## Incorporate any additional losses here (e.g. from regularizers)

            # Define optimizer and training function
            optimizer = optimizers.Adam()
            updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
            self.train_fn = K.function(
                inputs=[self.model.input, action_gradients, K.learning_phase()],
                outputs=[],
                updates=updates_op)
        
    class Critic:
        """Critic (Value) Model."""

        def __init__(self, state_size, action_size):
            """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
            """
            self.state_size = state_size
            self.action_size = action_size

            # Initialize any other variables here

            self.build_model()

        def build_model(self):
            """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
            # Define input layers
            states = layers.Input(shape=(self.state_size,), name='states')
            actions = layers.Input(shape=(self.action_size,), name='actions')

            # Add hidden layer(s) for state pathway
            net_states = layers.Dense(units=32, activation='relu')(states)
            net_states = layers.Dense(units=64, activation='relu')(net_states)

            # Add hidden layer(s) for action pathway
            net_actions = layers.Dense(units=32, activation='relu')(actions)
            net_actions = layers.Dense(units=64, activation='relu')(net_actions)

            ###### Try different layer sizes, activations, add batch normalization, regularizers, etc.

        
            # Combine state and action pathways
            net = layers.Add()([net_states, net_actions])
            net = layers.Activation('relu')(net)

            ###### Add more layers to the combined network if needed

        
            # Add final output layer to prduce action values (Q values)
            Q_values = layers.Dense(units=1, name='q_values')(net)

            # Create Keras model
            self.model = models.Model(inputs=[states, actions], outputs=Q_values)

            # Define optimizer and compile model for training with built-in loss function
            optimizer = optimizers.Adam()
            self.model.compile(optimizer=optimizer, loss='mse')

            # Compute action gradients (derivative of Q values w.r.t. to actions)
            action_gradients = K.gradients(Q_values, actions)

            # Define an additional function to fetch action gradients (to be used by actor model)
            self.get_action_gradients = K.function(
                inputs=[*self.model.input, K.learning_phase()],
                outputs=action_gradients)
       
    class OUNoise:
        """Ornstein-Uhlenbeck process."""

        def __init__(self, size, mu, theta, sigma):
            """Initialize parameters and noise process."""
            self.mu = mu * np.ones(size)
            self.theta = theta
            self.sigma = sigma
            self.reset()

        def reset(self):
            """Reset the internal state (= noise) to mean (mu)."""
            self.state = copy.copy(self.mu)

        def sample(self):
            """Update internal state and return it as a noise sample."""
            x = self.state
            dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
            self.state = x + dx
            return self.state
        
    class DDPG():
        """Reinforcement Learning agent that learns using DDPG."""
        def __init__(self, task):
            self.task = task
            self.state_size = task.state_size
            self.action_size = task.action_size
            self.action_low = task.action_low
            self.action_high = task.action_high
        
            # Actor (Policy) Model
            self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
            self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

            # Critic (Value) Model
            self.critic_local = Critic(self.state_size, self.action_size)
            self.critic_target = Critic(self.state_size, self.action_size)

            # Initialize target model parameters with local model parameters
            self.critic_target.model.set_weights(self.critic_local.model.get_weights())
            self.actor_target.model.set_weights(self.actor_local.model.get_weights())

            ######## Noise process //got ok result with 1.5 and 2
            self.exploration_mu = Exploration_Mu #default 0; long-term mean of the process
            self.exploration_theta = Exploration_Theta #default 0.15; rate of reversion to mean
            self.exploration_sigma = Exploration_Sigma #default 0.2; volatility of the Brownian motion
            self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

            # Replay memory
            self.buffer_size = Buffer_Size #default 100000
            self.batch_size = Batch_Size #default 64
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

            ######### Algorithm parameters
            self.gamma = DDPG_Gamma  # discount factor / default 0.99
            self.tau = DDPG_Tau      # for soft update of target parameters / default 0.01; 
                                     # higher means use more local# less target

        def reset_episode(self):
            self.noise.reset()
            state = self.task.reset()
            self.last_state = state
            return state

        def step(self, action, reward, next_state, done):
             # Save experience / reward
            self.memory.add(self.last_state, action, reward, next_state, done)

            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

            # Roll over last state and action
            self.last_state = next_state

        def act(self, state):
            """Returns actions for given state(s) as per current policy."""
            state = np.reshape(state, [-1, self.state_size])
            action = self.actor_local.model.predict(state)[0]
            return list(action + self.noise.sample())  # add some noise for exploration

        def learn(self, experiences):
            """Update policy and value parameters using given batch of experience tuples."""
            # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
            states = np.vstack([e.state for e in experiences if e is not None])
            actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
            rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
            dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
            next_states = np.vstack([e.next_state for e in experiences if e is not None])

            # Get predicted next-state actions and Q values from target models
            # Q_targets_next = critic_target(next_state, actor_target(next_state))
            actions_next = self.actor_target.model.predict_on_batch(next_states)
            Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

            # Compute Q targets for current states and train critic model (local)
            Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
            self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

            # Train actor model (local)
            action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
            self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

            # Soft-update target models
            self.soft_update(self.critic_local.model, self.critic_target.model)
            self.soft_update(self.actor_local.model, self.actor_target.model)   

        def soft_update(self, local_model, target_model):
            """Soft update model parameters."""
            local_weights = np.array(local_model.get_weights())
            target_weights = np.array(target_model.get_weights())

            assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

            new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
            target_model.set_weights(new_weights)
        
#### 1.3 Training the Agent Script
    
    # Containers to Hold Reward Data
    best_reward = -np.inf
    rewards = []
    
    # Labels to Write
    labels = ['episode','time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4',
              'reward','cumulative_reward']
    
    # If the File Output Already Exists than Remove it
    try:
        os.remove(File_Output)
    except OSError:
        pass
    
    # Write Data Labels to File
    with open(File_Output, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
         
    # Initiate the Agent and Task
    My_Task = Task(init_pose = Init_Pose,
                   init_velocities = Init_Velocities,
                   init_angle_velocities = Init_Angle_Velocities,
                   runtime = Runtime,
                   target_pos = Target_Pos)
    
    # Agent To Use
    Agent = DDPG(My_Task) 

    # Loop Through the Episode of Each Simulation Run
    for i_episode in range(1, Num_Episodes+1):
        # Reset The State and Cumulative Reward Per Each Episode
        state = Agent.reset_episode() 
        cumulative_reward = 0
    
        # Perform the simulation
        while True:
            action = Agent.act(state)
            
            # Adjust the Action per Action Size
            if Action_Size == 1:
                action_adjusted = action*4
            elif Action_Size == 2:
                action_adjusted = action*2
            elif Action_Size == 3:
                action_adjusted = (action*2)[:4]
            else:
                action_adjusted = action*1
                
            next_state, reward, done = My_Task.step(action_adjusted)
            Agent.step(action,reward,next_state, done)
            state = next_state
            cumulative_reward += reward
            
            # Capture the Rotor Speed as well.
            rotor_speeds = np.asarray(action_adjusted)
            
            # Write the Current Episode to CSV File
            to_write = [i_episode] + [My_Task.sim.time] + list(My_Task.sim.pose) + list(My_Task.sim.v) +\
            list(My_Task.sim.angular_v) + list(rotor_speeds)+[reward]+[cumulative_reward]
            
            # Open the Data File so that each simulation action can be saved
            with open(File_Output, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(to_write)
            
            # if the simulation is complete than record the cumulative reward
            # and compare it against the best reward so far.
            if done:
                if cumulative_reward > best_reward:
                    best_reward = cumulative_reward 
                rewards.append(cumulative_reward)
                print("\rEpisode = {:4d},     Cumulative_Reward = {:9.3f}     (Best = {:9.3f})".format(
                    i_episode, cumulative_reward, best_reward), end="")
                break
                sys.stdout.flush()  

    # Save The Results in a Dataframe
    Results = pd.read_csv(File_Output)
    print ('\nQuadcopter Trained!')
    return Results 