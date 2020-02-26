###############
# main.py: contains MDP and problem definitions
#
#  >> PLEASE NOTE THAT SOME QUESTION RESPONSES ARE OUT OF ORDER <<
#
# Name: Richard (Ruochen) Wang
# UID: 504770432
###############
# Preliminaries
# 0(a) Who did you collaborate with?
## No one else.
# 0(b)

# Import libraries
import numpy as np
import random

# State class,
## has ~ operator so that it can be indexed into an array directly
## ~State(x,y) == (y,x), where H is the height of the map
from state import State


###############
# Grid world construction
## We will need to reverse the order of the rows then transpose to be
# consistent with the indexing defined in the problem set
def form_grid(grid):
    return (np.flipud(grid)).T
    
## It is declared as depicted in the diagram under 2.2, not in index order
diagram  = np.array([[0,     0,      0,      0,      -100],
                     [0,     0,      0,      0,      -100],
                     [0,     np.nan, np.nan, 0,      -100],
                     [0,     0,      1,      0,      -100],
                     [0,     np.nan, np.nan, 0,      -100],
                     [0,     0,      10,     0,      -100]])

## form a array that can be index using (x,y)
grid_world = form_grid(diagram)
print(grid_world)
L_length = grid_world.shape[1]
W_width = grid_world.shape[0]

################
# 1(a) State space declaration
## The state space consists of all possible (x,y) pairs, stored as State objects
## The size of the state space is N_S = L_length * W_width
state_space = [State(x, y) for y in range(0, L_length) for x in range(0, W_width) ]
state_set = set(state_space)

################
# 1(b) Action space declaration
## The size of the action space is N_A = 5
action_space = { #'N': State(0, 0),   # Not
            'U': State(0, 1),   # Up
            'D': State(0, -1),  # Down
            'L': State(-1, 0),  # Left
            'R': State(1, 0) }  # Right
inv_action_space = {v: k for k, v in action_space.items()}

###############
# 3(a) Initial policy
init_policy = np.chararray(np.shape(grid_world))
init_policy[:] = 'L'
init_policy = np.char.decode(init_policy.tolist())

# 3(b) Display policy
def display_policy(policy):
    p_T = [list(i) for i in zip(*policy)]
    print(np.flipud(p_T))

# Markov decision process class wrapper
## Contains Markov decision process
class MarkovDP:
    def __init__(self, gamma, error_prob):
        self.gamma = gamma
        self.error_prob = error_prob
        
            
    ################
    # 1(c) and 2(a) State transition probability function, p_sa(s')
    ## inputs:                      returns:
    ## s:   current state           p_sa: size (H,L) array representing
    ## a:   action taken                  discrete probability distribution
    ## s':  outcome state
    def transition_prob_dist(self, current_state, action_taken):
        
        # initialize transition distribution as if 'N' is the action taken
        # equivalent to the transition distribution given a successful action 'N'
        TPD_success = np.zeros(np.shape(grid_world))
        TPD_success[~current_state] = 1
        
        if (action_taken == 'N'):
            return TPD_success
        
        # some action is taken
        elif (action_taken in action_space.keys()):
            # find possible directions to move in
            feasible_actions = self.get_feasible_actions(current_state)
                                    
            # set probability dist that we arrive at a some state given a failure event
            TPD_failure = np.zeros(np.shape(grid_world))
            ## accounts for 'N' when moving into invalid state or wall
            TPD_failure[~current_state] = 0.25 *  (len(action_space) - np.size(feasible_actions))
            
            for action in feasible_actions:
                adjacent_state = current_state + action
                TPD_failure[~adjacent_state] = 0.25
            
            # set probability dist that we arrive at a some state given a success event            
            desired_state = current_state + action_space[action_taken]
            if (self.get_reward(desired_state) in [np.nan]):
                TPD_success[~current_state] = 1
            else:
                TPD_success[~current_state] = 0
                TPD_success[~(current_state + action_space[action_taken])] = 1
            
            # return total probability combining error and success cases
            return (1 - self.error_prob) * TPD_success + self.error_prob * TPD_failure
        else:
            print(action_taken)
        
        return 0
    
    # iterates each through each possible action and returns with the set
    # of possible actions
    def get_feasible_actions(self, state):
        feasible_actions = []
        
        for action in action_space.values():
            adjacent_state = state + action
            
            if (not np.isnan(self.get_reward(adjacent_state))):
                feasible_actions.append(action)

        return feasible_actions
    
    ################
    # 2(b) get the reward value from the grid_world for state s
    # input:                      returns:
    # s:   state                  r:    reward value corresponding to the state
    #                                   returns nan if out of bounds
    def get_reward(self, state):
        if (state[0] < 0) or (state[1] < 0) or (state[0] > W_width-1) or (state[1] > L_length-1):
            return np.nan
        
        else:
            return grid_world[~state]
    
    ################
    # 3(c) policy evaluation, 
    ##     uses the Bellman equation to find the values under policy.
    ##     See reference 2., mdps.pdf
    # input:                      returns:
    # pi:  policy                 V^pi:  matrix of values
    def get_policy_eval(self, policy):
        rewards = grid_world.flatten()
        indices = ~np.isnan(rewards)
        n = np.count_nonzero(indices)
        
        P_ij = self.transition_prob_ss(n, policy)
                
        # calculate the values in the valid states, and fill the invalid ones with 0
        v = np.linalg.solve(np.eye(n) - self.gamma*P_ij, rewards[indices])
        V_pi = np.zeros([L_length*W_width,])
        V_pi[indices] = v
        V_pi.shape = [W_width,L_length]
        
        return V_pi
        
    def transition_prob_ss(self, size, policy):
        rewards = grid_world.flatten()
        indices = ~np.isnan(rewards)
        
        P_ij = np.zeros([size,size])
        
        itr = 0
        for a in range(0,len(state_space)):
            s = state_space[a]
            if (np.isnan(rewards[a])):
                continue
            
            PD_s = self.transition_prob_dist(s, policy[a % W_width][int(a/W_width)])
            
            P_ij[itr,:] = (PD_s.flatten())[indices]
            itr += 1
        
        return P_ij
    
    ################
    # 3(d) one step evaluation to find optimal policy
    def get_optimal_policy(self, value_distribution):
        rewards = grid_world.flatten()
        indices = ~np.isnan(rewards)
        n = np.count_nonzero(indices)
                
        opt_pol = ['N']*(W_width*L_length)
        
        # evaluate argmax policy at each s
        for a in range(0,np.size(state_space)):
            s = state_space[a]       
            if (np.isnan(rewards[a])):
                continue
                        
            r = rewards[a]
            opt_act = 'L'
                        
            feas_act = self.get_feasible_actions(s)
            for action in feas_act:
                movement = s + action
                r_p = rewards[movement[0]*L_length + movement[1]]
            
                if (r_p > r):
                    r = r_p
                    opt_act = inv_action_space[action]
            
            opt_pol[a] = opt_act
        
        return np.reshape(opt_pol,np.shape(grid_world))
    
    ################
    # 3(e) policy iteration
    def iterate_policy(self, policy, max_iter = 100):
        # initialize arbitarily
        val_func = np.zeros(np.shape(grid_world))
        
        rewards = grid_world.flatten()
        indices = ~np.isnan(rewards)
        n = np.count_nonzero(indices)
        
        opt_pol = policy
        
        # iterate and update
        for it in range(0, max_iter):
            v_pi_old = np.reshape(self.get_policy_eval(opt_pol), -1)
            opt_pol = self.get_optimal_policy(v_pi_old)
            P_ij = self.transition_prob_ss(n, opt_pol)
            
            v_pi = rewards[indices] + self.gamma * P_ij @ v_pi_old[indices]
            
            if (v_pi == v_pi_old):
                break
                
            V = np.zeros([L_length*W_width,])
            V[indices] = v_pi
            
        return (np.reshape(V,np.shape(grid_world)), opt_pol)
        
        
    
        
    
            
            



################ GENERATE REPORT ################
print("------- BEGIN REPORT -------")

print("1(a). See line XXXX for declaration. The size of the state space is LxH = 30.") #TODO
print("1(b). See line XXXX for declaration. The size of the action space is 5.") #TODO

print("\n")
print("----------------------------")

print("2(a). See line XXXX for declaration.")
print("2(b). See line XXXX for declaration.")

print("\n")
print("----------------------------")

print("3(a). See line XXXX for declaration.")
print("3(b). Displaying inital policy pi_0:")
display_policy(init_policy)
print("3(c). See line XXXX for declaration.")
print("3(d). See line XXXX for declaration. The one step imrpovement is given as")
mdp = MarkovDP(0.9,0.01)
pe = mdp.get_policy_eval(init_policy)
op = mdp.get_optimal_policy(pe)
display_policy(op)
print("3(e). Using 100 iterations or until convergece, find optimal policy")
import time
start = time.time()
opt_pol_V = mdp.iterate_policy(init_policy)
end = time.time()
display_policy(opt_pol_V[1])
print("3(f). The time is found to be " + str(end-start) + " seconds.")

print("\n")
print("----------------------------")