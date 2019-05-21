# Snake Environment
import pygame as pg
from scipy.misc import imread
#from pygame.locals import *
import random, math, os, shutil
import numpy as np

'''
Author: Nicolas-Reyland
my github: https://github.com/Nicolas-Reyland
first release on 12/02/2019
version: 0.2.1


This module is a personal open-source project
It contains a snake-game environment. The environment is made to look
the same as openai environments.

Check out openai here: https://github.com/openai
#I personally recommand yu the gym and retro repos

For this module, you will need the python-modules in requirements.txt
To install all the modules, open a terminal/cmd, go to the repo folder
and type: 'pip install -r requirements.txt'

If you read this, you can skip the README.md


Global Notes:
 - if you have any issue, tell me on github


'''

__id__ = 1

# ---------------------------------------------------------------- #
#                         Main Env Class                           #
# ---------------------------------------------------------------- #
class make(object):
    """
    This environment simulates Snake Game
    (it is an imitation of gym-library environments)

    * it has two main different behaviors:
     - 'maths' : the observation of a step (see after this paragraph) will be:
         - front bloacked (0 or 1)
         - right blocked (0 or 1)
         - left blocked (0 or 1)
         - angle with apple (from -1 to 1)
     - 'image' : the observation is the image of the game, like in gym-environments

    * you have 3 main methods:
     - reset
         - it resets the environment to first-time-called configuration
     - step
         - step has 1 input : the direction that the snake must move to: 'straight', 'left' or 'right'
         - step gives returns different data, depending on the observation_type (look above this paragraph)
     - render
         - render shows you the game (with pygame)

    * the subclasses:
     - the info subclass:
         - name: the environment name
         - id: the id of this environment, to track it through generations in a genetic algorithm, for example
         - observation_type: the envionment's observation_type ('mahts' or 'image')
         - env_path: the environment_path to store temporary images for image-observation_type
     - the action_space:
         - sample: a function to give a random action for step-function
         - shape: the length of the possible action-choices, for neural network output-layer, for example

    one more method is 'clear', to remove all temp-files and folders

    Notes:
     - you only can have the same env_path for multiple environments if you don't use those parallelly
     - if you use the 'image' observation_type, you should use the clean() method, else there would be remaining tmp-files
     - you can 'still' as an input for step (debugging only, normally)
     - the pygame coordinates start at the top-left corner and increment to the right and down. They're calculated in pixels

    """
    ### ----- environment initialisation -----
    def __init__(self, name='snake', id='auto', observation_type='image', env_path='snake_tmp'):
        # global pygame-values
        global cube_size, disp_w, disp_h
        # __id__'s purpose is auto-assignment of the envionment-id
        global __id__

        # auto-assignment of environment-id
        if id == 'auto':
            self.__id__ = __id__
            # increment for next envionment
            __id__ += 1
        else:
            # custom id
            self.__id__ = int(id)

        # notice environment-name is set automaticlly to 'snake'
        self.__name__ = name
        # observation_type is described in the env-class docstring
        self.__observation_type__ = observation_type

        # pygame-values
        cube_size, disp_w, disp_h = 20, 800, 600

        # __show__ makes the pygame-display update (changed with render())
        self.__show__ = False
        # if not, and render() is called, __init_pg__ is called to initialize pygame display and variables
        self.__pg_initialised__ = False
        # if observation_type is 'image', pygame must be used
        if self.__observation_type__ == 'image':
            # initialize pygame
            self.__init_pg__()
            self.__pg_initialised__ = True
            # set show to True
            self.__show__ = True

        ## --- Score and end of Game ---
        # Game is done?
        self.done = False
        # Total Score of last run
        self.score = 0
        # move_score is the reward of last step(). It is returned in step()
        self.__move_score__ = 0
        # default move_score (to start with aat each step())
        self.__default_move_score = 1
        # reward for an eaten apple
        self.__apple_eaten_reward__ = 3
        # malus if snake dies
        self.__dying_malus__ = -5

        ## --- game-variables ---
        # Snake initial coordinates
        self.__coords__ = [(disp_w/2-cube_size/2)//cube_size*cube_size, (disp_h-3*cube_size)//cube_size*cube_size]
        # is there an apple in the game ?
        self.__eaten__ = True
        # the snakeLength, incremented of 1 at every apple that gets eaten
        self.__snakeLength__ = 1
        # list of coordinates on pygame-display
        self.__snakeList__ = []
        # actual move, always start with 'up'
        self.__move__ =  'up'
        # apple coordinates
        self.__apple__ = None

        ## --- step() input variables ---
        # the current_direction is __current_direction__[1]
        self.__current_direction__ = ['left', 'up', 'right', 'down']
        # all possible move-choices
        self.__possible_choices__ = ['left', 'straight', 'right']

        ## --- handle observation_type ---
        # observation_type, described in the env-class docstring
        if self.__observation_type__ == 'maths':
            self.__image_observation__ = False
        elif self.__observation_type__ == 'image':
            self.__image_observation__ = True
        else:
            # only 'maths' and 'image' are availabale
            raise ValueError('Wrong Observation type {}. Avalaible: "maths" and "image"'.format(observation_type))

        ## --- environment_path ---
        # set environment path to ful-path
        self.__env_path__ = os.path.join(os.getcwd(), env_path)
        # only make the directory if it doesn't exit
        if not os.path.isdir(self.__env_path__) and self.__image_observation__:
            os.mkdir(self.__env_path__)

        ## --- initialisation of sub_classes ---
        self.info = __info__(self.__name__, self.__id__, self.__observation_type__, self.__env_path__)
        self.action_space = __action_space__(self)

    ### ----- "one time" environment functions -----
    def clean(self):
        # remove the tmp-folder and all the tmp-files of envionment
        shutil.rmtree(self.__env_path__)

    def __init_pg__(self):
        # initialisation of pygame
        pg.init()
        # set screen (display) variable
        self.screen = pg.display.set_mode((disp_w,disp_h))
        # set the window caption
        pg.display.set_caption('Snake')

    def reset(self):
        # reset class with first-time given arguments
        self.__init__(name=self.info.name,
                        id=self.info.id,
                        observation_type=self.info.observation_type,
                        env_path=self.__env_path__)

    ### ----- graphics -----
    def render(self):
        # __show__ is always set to False after a step, so if
        # you want to see the game, render will set it to True
        if not self.__pg_initialised__:
            self.__init_pg__()
        self.__show__ = True

    ### ----- hidden environment functions -----
    def __rotate__(self, l, n):
        # rotate 1D array
        return l[n:] + l[:n]

    def __eval_action__(self, action):
        # 'still' action is for debugging, but I permit you to use it with step()
        if action == 'still':
            # 'still' makes the snake stay static (doesn't move)
            return action
        if action == 'straight':
            # stay as it is
            pass
        elif action == 'left':
            # l, >u<, r, d => d, >l<, u, r
            self.__current_direction__ = self.__rotate__(self.__current_direction__, -1)
        elif action == 'right':
            # l, >u<, r, d => u, >r<, d, l
            self.__current_direction__ = self.__rotate__(self.__current_direction__, 1)
        # as said in the __init__, the current direction of the snake is __current_direction__[1]
        return self.__current_direction__[1]

    ### ----- snake-game "action" function -----
    def step(self, move):
        # global values for pygame
        global cube_size, disp_w, disp_h

        # the move comes as described in the env-class docstring, but an sort-of euclidean direction is needed
        move = self.__eval_action__(move) # returns a directions (see the if-statements beneath)

        # initial move_score (reward) is 1
        self.__move_score__ = self.__default_move_score

        # change the coordinates of the snake in correlation with the chosen direction
        if move == 'right': self.__coords__[0] += cube_size
        elif move == 'left': self.__coords__[0] -= cube_size
        elif move == 'up': self.__coords__[1] -= cube_size
        elif move == 'down': self.__coords__[1] += cube_size
        elif move == 'still': pass # not needed, normally. only for debugging
        else: raise ValueError('{} is not a valid move!'.format(move))

        # snakeHead is always initialized to an empty list
        snakeHead = []
        # add the new coordinates to snakeHead
        snakeHead.append(self.__coords__[0])
        snakeHead.append(self.__coords__[1])
        # add the head-of-snake to snakeList
        self.__snakeList__.append(snakeHead)

        # reduce the snakeList in relation-ship with snakeLength
        if len(self.__snakeList__) > self.__snakeLength__:
            del self.__snakeList__[0]

        # if __eaten__, generate new random apple coordinates
        if self.__eaten__:
            self.__apple__ = [random.randrange(0,disp_w-cube_size,cube_size), random.randrange(0,disp_h-cube_size,cube_size)]
            self.__eaten__ = False

        # fill the screen with white color
        if self.__show__: self.screen.fill((255,255,255))

        # if the snake goes outside the pygame-window, done = True
        if self.__coords__[0] < 0 or self.__coords__[0]+cube_size > disp_w or self.__coords__[1] < 0 or self.__coords__[1]+cube_size > disp_h:
            self.done = True

        # if snake is at same position than apple
        if self.__coords__ == self.__apple__:
            # increment snakeLength
            self.__snakeLength__ += 1
            # eaten=True, so next frame there will be a new random-generate apple
            self.__eaten__ = True
            # move_score (reward) set to apple_eaten_reward if an apple has been eaten
            self.__move_score__ = self.__apple_eaten_reward__

        # draw the apple (red coor)
        if self.__show__: pg.draw.rect(self.screen, (210,0,0), (self.__apple__[0], self.__apple__[1], cube_size, cube_size))

        # snake colliding with itself
        for bodypart in self.__snakeList__[:-1]:
            if bodypart == snakeHead:
                self.done = True

        # draw a the snakeHead
        if self.__show__:
            pg.draw.rect(self.screen, (0,155,0), (self.__coords__[0], self.__coords__[1], cube_size, cube_size))
            # update the pygame-display
            pg.display.update()
            #pg.time.wait(120)

        # if game is done, add the malus (normally negative number)
        if self.done:
            self.__move_score__ = self.__dying_malus__

        # calculate score
        self.score += self.__move_score__

        # __show__ is true if env.render() is called or if observation_type is 'image'
        if self.__observation_type__ != 'image':
            self.__show__ = False

        if self.__image_observation__:
            # create an image with pygame.save function and save it to environment_path (tmp-folder)
            image = pg.image.save(self.screen, os.path.join(self.__env_path__, 'tmp_image.png'))
            # read the image (gives a numpy array) with imread (from scipy.misc)
            image_data = imread(os.path.join(self.__env_path__, 'tmp_image.png'))
            # observation, reward, done, info
            return image_data, self.__move_score__, self.done, (snakeHead, self.__snakeList__, self.__apple__)
        # the view_list is the "block-blocked or not" list
        viewlist = __get_view_list__(self.__move__, snakeHead, self.__snakeList__, self.__apple__)
        # unpack these variables from the viewlist
        ahead, right, left = viewlist#[0], viewlist[1], viewlist[2]
        # observation, reward, done, info
        return (ahead, right, left, __angle_with_apple__(snakeHead, self.__apple__)), self.__move_score__, self.done, (snakeHead, self.__snakeList__, self.__apple__)


### ----- side classes ----
class __info__:
    '''
    to see more information about __info__ class,
    see the env-class docstring about its info-class

    Note:
     Those class' attributes are static and won't syncronize
     with the parent-env-class. To syncronize, you can do it manually
     or by resetting the parent environment.

    '''
    ### ----- sub "info" class initialisation -----
    def __init__(self, name, id, observation_type, path):
        self.name = name
        self.id = id
        self.observation_type = observation_type
        self.env_path = path


class __action_space__:
    '''
    to see more information about __action_space__ class,
    see the env-class docstring about its action_space-class

    Note:
     This sub-class is snchronized with the parent-env-class

    '''
    ### ----- information about game-actions -----
    def __init__(self, base):
        # base is the environment class this __action_space__ class belongs to
        self.__base__ = base
        # the shape of a possible neural network output
        self.shape = len(self.__base__.__possible_choices__)

    def sample(self):
        # return a random sample of avalaible choices (all except going backwards)
        return random.choice(self.__base__.__current_direction__[:-1]) # the last one is going back, impossible in snake

# -------------------------------- #
# ----- Snake Side-Functions ----- #
# -------------------------------- #
## --- angle_with_apple is NOT from me: https://theailearner.com/2018/04/19/snake-game-with-deep-learning/ ---
def __angle_with_apple__(snake_position, apple_position):
    apple_direction_vector = np.array(apple_position)-np.array(snake_position[0])
    snake_direction_vector = np.array(snake_position)-np.array(snake_position[1])
    #                                 snake_position[0]

    norm_of_apple_direction_vector = np.linalg.norm(apple_direction_vector)
    norm_of_snake_direction_vector = np.linalg.norm(snake_direction_vector)
    if norm_of_apple_direction_vector == 0:
        norm_of_apple_direction_vector = 10
    if norm_of_snake_direction_vector == 0:
        norm_of_snake_direction_vector = 10

    apple_direction_vector_normalized = apple_direction_vector/norm_of_apple_direction_vector
    snake_direction_vector_normalized = snake_direction_vector/norm_of_snake_direction_vector
    angle = math.atan2(apple_direction_vector_normalized[1] * snake_direction_vector_normalized[0] - apple_direction_vector_normalized[0] * snake_direction_vector_normalized[1], apple_direction_vector_normalized[1] * snake_direction_vector_normalized[1] + apple_direction_vector_normalized[0] * snake_direction_vector_normalized[0]) / math.pi
    return angle

## --- get the elements from
def __get_view_list__(direction, snakeHead, snakeList, apple):
    view_list = []
    if direction == 'up':
        view_list.append(any(bodypart[1] < snakeHead[1] for bodypart in snakeList) and snakeHead[1] != 0) # ahead
        view_list.append(any(bodypart[0] > snakeHead[0] for bodypart in snakeList)) # right
        view_list.append(any(bodypart[0] < snakeHead[0] for bodypart in snakeList)) # left
        view_list.append(apple[1] < snakeHead[1])
        view_list.append(apple[0] < snakeHead[0])
        view_list.append(apple[0] > snakeHead[0])
    elif direction == 'down':
        view_list.append(any(bodypart[1] > snakeHead[1] for bodypart in snakeList) and snakeHead[1]+cube_size != disp_h) # ahead
        view_list.append(any(bodypart[0] < snakeHead[0] for bodypart in snakeList)) # right
        view_list.append(any(bodypart[0] > snakeHead[0] for bodypart in snakeList)) # left
        view_list.append(apple[1] > snakeHead[1])
        view_list.append(apple[0] > snakeHead[0])
        view_list.append(apple[0] < snakeHead[0])
    elif direction == 'right':
        view_list.append(any(bodypart[0] > snakeHead[0] for bodypart in snakeList) and snakeHead[0] != disp_w-cube_size) # ahead
        view_list.append(any(bodypart[1] > snakeHead[1] for bodypart in snakeList)) # right
        view_list.append(any(bodypart[1] < snakeHead[1] for bodypart in snakeList)) # left
        view_list.append(apple[0] < snakeHead[0])
        view_list.append(apple[1] < snakeHead[1])
        view_list.append(apple[1] > snakeHead[1])
    elif direction == 'left':
        view_list.append(any(bodypart[0] < snakeHead[0] for bodypart in snakeList) and snakeHead != 0)# ahead
        view_list.append(any(bodypart[1] < snakeHead[1] for bodypart in snakeList)) # right
        view_list.append(any(bodypart[1] > snakeHead[1] for bodypart in snakeList)) # left
        view_list.append(apple[0] > snakeHead[0])
        view_list.append(apple[1] > snakeHead[1])
        view_list.append(apple[1] < snakeHead[1])
    viewlist = []
    for i in view_list:
        if i: viewlist.append(1)
        else: viewlist.append(0)
    return viewlist
