import numpy as np
from gymnasium import spaces
import gymnasium as gym


class SnakeGame(gym.Env):
    # other metadata avail, render.moldes unncessary if render() is not implemented
    metadata = {'render.modes' : ['console', 'rgb_array']}

    n_actions = 3

    # actions
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2

    # states
    EMPTY = 0
    SNAKE = 1
    WALL = 2
    FOOD = 3

    REWARD_WALL_HIT = -50
    REWARD_PER_STEP_TOWARDS_FOOD = 5 # avoid hitting walls on purpose
    REWARD_PER_FOOD = 100
    MAX_STEPS_AFTER_FOOD = 200 # avoid loop


    def grid_distance(self, pos1, pos2):
        # calculate euclidean distance between 2 points
        return np.linalg.norm(np.array(pos1, dtype=np.float32) - np.array(pos2, dtype=np.float32))

    
    def __init__(self, grid_size=20):
        super(SnakeGame, self).__init__()

        # steps init
        self.stepnum = 0
        self.last_food_step = 0

        # grid init
        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8) + self.EMPTY # EMPTY is zero so it doesn't matter (in case its not)
        
        # wall init
        self.grid[0, :] = self.WALL # UP
        self.grid[:, 0] = self.WALL # LEFT
        self.grid[self.grid_size - 1, :] = self.WALL # DOWN
        self.grid[:, self.grid_size - 1] = self.WALL # RIGHT

        # snake init
        # self.snake_coordinates = [ (1,1), (2,1) ]
        self.snake_coord = [(4, 3), (4, 4)] # top left

        for coord in self.snake_coord:
            self.grid[coord] = self.SNAKE

        # food init
        self.grid[3, 3] = self.FOOD

        # distance calculation
        self.head_dist_to_food = self.grid_distance(
            self.snake_coord[-1],
            np.argwhere(self.grid == self.FOOD)[0]
        )

        # save init setup
        self.init_grid = self.grid.copy()
        self.init_snake_coord = self.snake_coord.copy()

        # action space
        self.action_space = spaces.Discrete(self.n_actions)

        # observation(state) space
        self.observation_space = spaces.Dict(
            spaces={
                "position" : spaces.Box(low=0, high=(self.grid_size - 1), shape=(2,), dtype=np.int32),
                "direction" : spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32),
                "grid" : spaces.Box(low=0, high=3, shape=(self.grid_size * self.grid_size,), dtype=np.uint8)
            }
        )
    
    
    def reset(self, seed=None):
        import random
        # to init position
        self.stepnum = 0
        self.last_food_step = 0
        self.grid = self.init_grid.copy()
        self.snake_coord = self.init_snake_coord.copy()

        self.head_dist_to_food = self.grid_distance(
            self.snake_coord[-1],
            np.argwhere(self.grid == self.FOOD)[0]
        )

        if seed is not None:
            np.random.seed(seed)
        
        obs = self._get_obs() # state space
        info = {}

        return obs, info


    def _get_obs(self):
        position = np.array(self.snake_coord[-1], dtype=np.int32)
        direction = (np.array(self.snake_coord[-1]) - np.array(self.snake_coord[-2])).astype(np.int32)
        grid = self.grid.flatten()

        obs = {
            "position" : position,
            "direction" : direction,
            "grid" : grid
        }
        
        return obs
    

    def step(self, action):
        direction = np.array(self.snake_coord[-1]) - np.array(self.snake_coord[-2])

        if action == self.STRAIGHT:
            step = direction # towards the direction the snake faces
        elif action == self.RIGHT:
            # rotation matrix
            step = np.array( [direction[1], -direction[0]] )
        elif action == self.LEFT:
            step = np.array( [-direction[1], direction[0]] )
        
        # new head
        new_coord = (np.array(self.snake_coord[-1]) + step).astype(np.int32)

        if not (0 <= new_coord[0] < self.grid_size and 0 <= new_coord[1] < self.grid_size):
            return self._get_obs(), self.REWARD_WALL_HIT, True, False, {}

        self.snake_coord.append( (new_coord[0], new_coord[1]) )

        new_pos = self.snake_coord[-1]
        new_pos_type = self.grid[new_pos]
        self.grid[new_pos] = self.SNAKE

        done = False
        reward = 0 # calculated later

        if new_pos_type == self.FOOD:
            reward += self.REWARD_PER_FOOD
            self.last_food_step = self.stepnum

            # new food
            empty_tiles = np.argwhere(self.grid == self.EMPTY)

            if len(empty_tiles):
                new_food_pos = empty_tiles[np.random.randint(0, len(empty_tiles))]
                self.grid[new_food_pos[0], new_food_pos[1]] = self.FOOD
            else:
                done = True
            
        else:
            self.grid[self.snake_coord[0]] = self.EMPTY # empty the snake tail
            self.snake_coord = self.snake_coord[1:]

            if (new_pos_type == self.WALL) or (new_pos_type == self.SNAKE):
                done = True
                reward += self.REWARD_WALL_HIT
        
        head_dist_to_food_prev = self.head_dist_to_food
        self.head_dist_to_food = self.grid_distance(
            self.snake_coord[-1],
            np.argwhere(self.grid == self.FOOD)[0]
        )

        # reward for distance between snake <-> food
        if head_dist_to_food_prev > self.head_dist_to_food:
            reward += self.REWARD_PER_STEP_TOWARDS_FOOD
        elif head_dist_to_food_prev < self.head_dist_to_food:
            reward -= self.REWARD_PER_STEP_TOWARDS_FOOD * 2
        
        # max steps since no food
        if ((self.stepnum - self.last_food_step) > self.MAX_STEPS_AFTER_FOOD):
            done = True
        
        self.stepnum += 1

        # print(f"Step: {self.stepnum}, Action: {action}, Reward: {reward}, Done: {done}")
        # print(f"Snake Head: {self.snake_coord[-1]}, Distance to Food: {self.head_dist_to_food}")
        # print(f"New Position Type: {new_pos_type}")

        # return observation, reward, done, truncated, info
        return self._get_obs(), reward, done, False, {}


    def snake_plot(self, plot_inline=False):
        wall_idx = (self.grid == self.WALL)
        snake_idx = (self.grid == self.SNAKE)
        food_idx = (self.grid == self.FOOD)

        # colour array for plot
        colour_arr = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8) + 255 # default to white
        colour_arr[wall_idx, :] = np.array([0, 0, 0])
        colour_arr[snake_idx, :] = np.array([255, 196, 0])
        colour_arr[food_idx, :] = np.array([30, 47, 135])

        return colour_arr
    

    def render(self, mode='rgb_array'):
        if mode == 'console':
            print(self.grid)
        elif mode == 'rgb_array':
            return self.snake_plot()
