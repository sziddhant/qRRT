import sys
import numpy as np

from tqdm import tqdm
from actor import Actor
from critic import Critic
from utils.stats import gather_stats
from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from tensorflow.python.ops.gen_summary_ops import summary_writer
from utils.memory_buffer import MemoryBuffer
import random, math, pygame,time
from pygame.locals import *
from math import sqrt, cos, sin, atan2, floor
from rrt import env_step,env_reset

##################################################################################### RRT ##########################################################################################

goal_pos = (330, 80)
init_pos = (400, 450)
path = []
# constants
XDIM = 720
YDIM = 500
WINSIZE = [XDIM, YDIM]
EPSILON = 20.0
NUMNODES = 10000
GOAL_RADIUS = 10
# MIN_DISTANCE_TO_ADD = 1.0
GAME_LEVEL = 3
pygame.init()
fpsClock = pygame.time.Clock()

# initialize and prepare screen
screen = pygame.display.set_mode(WINSIZE)
pygame.display.set_caption('Rapidly Exploring Random Tree')
white = 255, 240, 200
black = 20, 20, 40
red = 255, 0, 0
blue = 0, 255, 0
green = 0, 0, 255
cyan = 0, 255, 255

# setup program variables
count = 0
rectObs = []
path=[]
pathAct=[]
pathR=[]
nodes = []
nodesA = []



class Node(object):
    """Node in a tree"""

    def __init__(self, point, parent):
        super(Node, self).__init__()
        self.point = point
        self.parent = parent


def angle(a, b):
    return atan2(b[1]-a[1], b[0]-a[0])


def angle2(p1, p2):
    return atan2(p1[1]-p2[0], p1[1]-p2[1])


def dist(p1, p2):
    return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def point_circle_collision(p1, p2, radius):
    distance = dist(p1, p2)
    if (distance <= radius):
        return True
    return False


def step_from_to_new(p,theta):
    return (p[0] + EPSILON * cos(theta)), (p[1] + EPSILON * sin(theta))


def step_from_to(p1, p2):
    theta = atan2(p2[1] - p1[1], p2[0] - p1[0])
    return (p1[0] + EPSILON * cos(theta)), (p1[1] + EPSILON * sin(theta))


def collides(p):
    global rectObs
    for rect in rectObs:
        if rect.collidepoint(p) == True:
            # print ("collision with object: " + str(rect))
            return True
    return False


def get_random():
    return random.random() * XDIM, random.random() * YDIM


def get_random_clear():
    while True:
        p = get_random()
        noCollision = collides(p)
        if noCollision == False:
            return p


def init_obstacles(configNum):
    global rectObs
    rectObs = []
    print("config " + str(configNum))
    if (configNum == 0):
        rectObs.append(pygame.Rect((XDIM / 2.0 - 50, YDIM / 2.0 - 100), (100, 200)))
    if (configNum == 1):
        rectObs.append(pygame.Rect((40, 10), (100, 200)))
        rectObs.append(pygame.Rect((500, 200), (500, 200)))
    if (configNum == 2):
        rectObs.append(pygame.Rect((40, 10), (100, 200)))
    if (configNum == 3):
        rectObs.append(pygame.Rect((40, 10), (100, 200)))
        rectObs.append(pygame.Rect((0, 500), (-1, 500)))
        rectObs.append(pygame.Rect((720, 500), (750, 500)))
        rectObs.append(pygame.Rect((0, 0), (750, 0)))
        rectObs.append(pygame.Rect((0, 500), (750, 500)))

    for rect in rectObs:
        pygame.draw.rect(screen, red, rect)


def reset():
    global count
    screen.fill(black)
    init_obstacles(GAME_LEVEL)
    count = 0


def full_rrt(k):
    global count
    global init_pos
    global goal_pos
    global nodes
    global path
    global GOAL_RADIUS
    global GAME_LEVEL
    nodes=[]
    path=[]
    initPoseSet = True
    goalPoseSet = True
    goalPoint = Node(None, None)
    currentState = 'buildTree'
    initialPoint = Node(init_pos, None)
    nodes.append(initialPoint)
    init_obstacles(GAME_LEVEL)
    pygame.draw.circle(screen, blue, initialPoint.point, GOAL_RADIUS)
    goalPoint = Node(goal_pos, None)
    pygame.draw.circle(screen, green, goalPoint.point, GOAL_RADIUS)

    while k > 0:
        if currentState == 'goalFound':
            # traceback
            currNode = goalNode
            # print (currNode.point)
            while currNode.parent != None:
                pygame.draw.line(screen, cyan, currNode.point, currNode.parent.point)
                done = False
                if(currNode == goalNode):
                    done = True
                r = -1 * sqrt((currNode.point[0]-goal_pos[0])*(currNode.point[0]-goal_pos[0])+((currNode.point[1]-goal_pos[1])*(currNode.point[1]-goal_pos[1])))
                path.append((currNode.parent.point, currNode.point, r, done))
                pathAct.append(angle2(currNode.parent.point, currNode.point))
                currNode = currNode.parent
            pygame.display.update()
            time.sleep(0.25)
            k = k - 1
            reset()
            optimizePhase = True
        elif currentState == 'optimize':
            fpsClock.tick(5.5)
            pass
        elif currentState == 'buildTree':
            count = count + 1
            if count < NUMNODES:
                foundNext = False
                while foundNext == False:
                    rand = get_random_clear()
                    # print("random num = " + str(rand))
                    parentNode = nodes[0]
                    for p in nodes:  # find nearest vertex
                        if dist(p.point, rand) <= dist(parentNode.point,rand):  # check to see if this vertex is closer than the previously selected closest
                            newPoint = step_from_to(p.point, rand)
                            if collides(newPoint) == False:  # check if a collision would occur with the newly selected vertex
                                parentNode = p  # the new point is not in collision, so update this new vertex as the best
                                foundNext = True

                newnode = step_from_to(parentNode.point, rand)
                nodes.append(Node(newnode, parentNode))
                pygame.draw.line(screen, white, parentNode.point, newnode)

                if point_circle_collision(newnode, goalPoint.point, GOAL_RADIUS):
                    currentState = 'goalFound'
                    goalNode = nodes[len(nodes) - 1]
                    print (goalNode.point)

            else:
                print("Ran out of nodes... :(")

        # handle events
        pygame.event.get()
        pygame.display.update()
        fpsClock.tick(10000)
    return (path,pathAct)


##################################################################################### RRT ##########################################################################################


class DDPG:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, state_dim, act_range, k, buffer_size = 20000, nb_episodes=5000, gamma = 0.99, lr = 0.0005, tau = 0.001, batch_size = 128):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.state_dim = state_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size=batch_size
        self.nb_episodes = nb_episodes
        # Create actor and critic networks
        self.actor = Actor(self.state_dim, act_dim, act_range, 0.1 * lr, tau)
        self.critic = Critic(self.state_dim, act_dim, lr, tau)
        self.buffer = MemoryBuffer(buffer_size)

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)[0]

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self):
        results = []

        # First, gather experience
        tqdm_e = tqdm(range(self.nb_episodes), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            noise = OrnsteinUhlenbeckProcess(size=self.act_dim)
            nb_steps=0
            init_obstacles(GAME_LEVEL)
            pygame.draw.circle(screen, blue, init_pos, GOAL_RADIUS)
            pygame.draw.circle(screen, green, goal_pos, GOAL_RADIUS)
            while(nb_steps<self.batch_size):
                p = get_random_clear()
                a = self.policy_action(p)[0]
                #a = np.clip(a + noise.generate(time), -self.act_range, self.act_range)
                p1 = step_from_to_new(p,a)
                pygame.event.get()
                pygame.draw.line(screen, white, p, p1)
                pygame.display.update()
                pygame.event.get()
                reward = -0.01*((p1[0] - goal_pos[0]) * (p1[0] - goal_pos[0]) + (p1[1] - goal_pos[1]) * (p1[1] - goal_pos[1]) - (p[0] - goal_pos[0]) * (p[0] - goal_pos[0]) + (p[1] - goal_pos[1]) * (p[1] - goal_pos[1]))
                done = False
                if(collides(p1)):
                    reward = -1000
                if(point_circle_collision(p1, goal_pos, GOAL_RADIUS)):
                    done = True
                    reward=1000
                self.memorize(p, a, reward, done, p1)
                nb_steps+=1
            states, actions, rewards, dones, new_states, _ = self.sample_batch(self.batch_size)
            a_new = self.actor.target_predict(new_states)
            #print(a_new)
            q_values = self.critic.target_predict([new_states, a_new])
            # Compute critic target
            critic_target = self.bellman(rewards, q_values, dones)
            # Train both networks on sampled batch, update target networks
            self.update_models(states, actions, critic_target)
            cumul_reward += reward
            time += 1
            '''state, action = full_rrt(1);
            # Add outputs to memory buffer
            for i in range(len(state)):
                self.memorize(state[i][0], action[i], state[i][2], state[i][3], state[i][1])
            # Sample experience from buffer
            states, actions, rewards, dones, new_states, _ = self.sample_batch(self.batch_size)
            # Predict target q-values using target networks
            q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
            # Compute critic target
            critic_target = self.bellman(rewards, q_values, dones)
            # Train both networks on sampled batch, update target networks
            self.update_models(states, actions, critic_target)
            cumul_reward += r
            time += 1

            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])'''

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            #summary_writer.add_summary(score, global_step=e)
            #summary_writer.flush()
            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()
            reset()
        #print(a_new)
        print('################################')
        old_state = env_reset()
        reset()
        noise = OrnsteinUhlenbeckProcess(size=self.act_dim)
        time, cumul_reward, done = 0, 0, False
        batch_size = 128

        while not done:
            # zz-=1
            pygame.display.update()
            pygame.event.get()

            # Actor picks an action (following the deterministic policy)
            a = self.policy_action(old_state)
            print(a)
            # Clip continuous values to be valid w.r.t. environment
            #a = np.clip(a , -self.act_range, self.act_range)
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env_step(a)
            # Add outputs to memory buffer
            #self.memorize(old_state, a, r, done, new_state)
            # Sample experience from buffer
            #states, actions, rewards, dones, new_states, _ = self.sample_batch(batch_size)
            # Predict target q-values using target networks
            # print(states)
            #q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
            # Compute critic target
            #critic_target = self.bellman(rewards, q_values, dones)
            # Train both networks on sampled batch, update target networks
            #self.update_models(states, actions, critic_target)
            # Update current state
            old_state = new_state
            cumul_reward += r
            time += 1
        return results

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)


algo = DDPG(1, (2,), 3.14, 0)
algo.train()
