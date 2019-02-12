# Example for Snake Environment
import snake
# I'll use opencv for this example
import cv2

'''
This is the example.py to show a basic example
with snake environment

'''
# create the environment. notice all these arguments are optional
env = snake.make(name='snake_example', id='auto', observation_type='image', env_path='snake_temporary_file')
# you don't need to do a env.reset()

# print some information about the envionment
print('Some information about the environment (in "info" sub-class)')
print('name: {}'.format(env.info.name))
print('id: {}'.format(env.info.id))
print('observation: {}'.format(env.info.observation_type))
print('environment path: {}'.format(env.info.env_path))

# get the action_space information
print('\nAction Space information:')
print('Action Shape is {}'.format(env.action_space.shape))














#
