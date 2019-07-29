import gym
from gym import error, spaces, utils
from gym.utils import seeding
from math import sqrt,cos,sin,atan2
import sys, random, math, pygame
import time
from pygame.locals import *
import random
import numpy as np

#constants
XDIM = 720
YDIM = 500
WINSIZE = [XDIM, YDIM]
EPSILON = 15.0
NUMNODES = 5000
GOAL_RADIUS = 30
MIN_DISTANCE_TO_ADD = 1.0
GAME_LEVEL = 1
goalPoint = (600,100)
initialPoint = (600,450)

pygame.init()
fpsClock = pygame.time.Clock()

#initialize and prepare screen
screen = pygame.display.set_mode(WINSIZE)
pygame.display.set_caption('Rapidly Exploring Random Tree')
white = 255, 240, 200
black = 20, 20, 40
red = 255, 0, 0
blue = 0, 255, 0
green = 0, 0, 255
cyan = 0,255,255

# setup program variables
count = 0
rectObs = []


# learning variables

isFirst = 1
initVal = 0
gamma = 0.9
currentState="buildTree"

def dist(p1,p2):
    return sqrt(1*(p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def point_circle_collision(p1, p2, radius):
    distance = dist(p1,p2)
    #print (distance-radius)
    if (distance <= radius):
        return True
    return False



def init_obstacles(configNum):
    global rectObs
    rectObs = []
    #print("config "+ str(configNum))
    if (configNum == 0):
        rectObs.append(pygame.Rect((XDIM / 2.0 - 50, YDIM / 2.0 - 100),(100,200)))
    if (configNum == 1):
        rectObs.append(pygame.Rect((40,10),(100,200)))
        rectObs.append(pygame.Rect((500,200),(500,200)))
    if (configNum == 2):
        rectObs.append(pygame.Rect((40,10),(100,200)))
    if (configNum == 3):
        rectObs.append(pygame.Rect((40,10),(100,200)))

    for rect in rectObs:
        pygame.draw.rect(screen, red, rect)

def collides(p):
    global rectObs
    for rect in rectObs:
        if rect.collidepoint(p) == True:
            # print ("collision with object: " + str(rect))
            return True
    return False
def reset():
    global count
    global path
    global flag
    screen.fill(black)
    init_obstacles(GAME_LEVEL)
    count = 0


'''class WrapperBox(spaces.Box):
	def __init__(self, low, high, shape=None, dtype=np.float32):
		super(WrapperBox, self).__init__(low, high, shape=None, dtype=np.float32)

	def sample(self):
		p = get_random_clear()
		return p'''

class qrrtEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		global initialPoint
		global GOAL_RADIUS
		self.state = initialPoint
		self.done = 0
		self.reward = 0
		self.action_space=spaces.Discrete(4)
		self.observation_space=spaces.Box(np.array([0,0]), np.array([XDIM, YDIM]), dtype=np.float32)
		screen.fill(black)
		init_obstacles(GAME_LEVEL)
		pygame.draw.circle(screen, blue, initialPoint, GOAL_RADIUS)
		pygame.draw.circle(screen, green, goalPoint, GOAL_RADIUS)
		pygame.display.update()


	def step(self, action):
		angle=0
		if(action==0):
			angle=0
		elif(action==1):
			angle=math.pi/2
		elif(action==2):
			angle=math.pi
		elif(action==3):
			angle=3*(math.pi/2)
		else:
			print(action)
			angle=-math.pi/2
		new_state = (self.state[0]+EPSILON*cos(angle), self.state[1]+EPSILON*sin(angle))
		done=False
		#reward=-dist(new_state, goalPoint)
		reward = -1
		if collides(new_state)or (XDIM<new_state[0])or(new_state[0]<0) or (YDIM<new_state[1])or (new_state[1]<0):
			reward-=1000
			#print("!!")
			new_state=self.state
			#done = True
		
		#reward =-1
		if(point_circle_collision(new_state, goalPoint, GOAL_RADIUS)):
			done=True
			reward+=10000
		pygame.draw.line(screen, white, self.state, new_state)
		self.state = new_state
		pygame.event.get()
		#print(new_state)
		#print (reward)
		return (new_state, reward, done, {})


	def reset(self):
		self.state = initialPoint
		self.reward = 0
		screen.fill(black)
		init_obstacles(GAME_LEVEL)
		pygame.draw.circle(screen, blue, initialPoint, GOAL_RADIUS)
		pygame.draw.circle(screen, green, goalPoint, GOAL_RADIUS)
		pygame.display.update()
		return self.state

  
	def render(self, mode='human', close=False):
		pygame.display.update()
		pygame.event.get()
		pygame.display.update()
