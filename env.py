import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
import matplotlib.pyplot as plt

class GridEnv(gym.Env):
    def __init__(self, render_mode=None, device='cuda'):
        super(GridEnv, self).__init__()

        self.grid_size = 6  # grid size
        self.action_space = spaces.Discrete(4)  # 4 possible actions (up, down, left, right)
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)  # 2D position
        self.render_mode = render_mode
        self.device = device

        # Grid definition

        self.grid = th.tensor([
            [0, 1, 2, 0, 0, 2],
            [0, 2, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [0, 2, 1, 0, 2, 0], 
            [0, 1, 1, 0, 1, 0]
        ], dtype=th.float32, device=self.device)

        self.pheromones = th.ones_like(self.grid, dtype=th.float32, device=self.device)*0.1 # Initial pheromone level
        self.max_pheromone = 5.0  # Maximum pheromone level

        # Increase pheromone levels at mandatory points
        mandatory_points = (self.grid == 2).nonzero(as_tuple=True)
        self.pheromones[mandatory_points] = 1.0  # Set higher initial pheromone level at mandatory points
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = th.tensor([0, 0], device=self.device, dtype=th.int32)  # Start at bottom left corner
        self.pheromones = th.ones_like(self.grid, dtype=th.float32, device=self.device)*0.1
        self.vistited_mandatory_pts = set()
        print("Environment reset. Initial state:", self.state.cpu().numpy())
        return self.state.cpu().numpy(), {}

    def step(self, action):
        # Movement directions
        directions = th.tensor([
            [-1, 0],  # Up
            [1, 0],   # Down
            [0, -1],  # Left
            [0, 1]    # Right
        ], device=self.device, dtype=th.int32)

        # Calculation of new position based on previous action
        new_position = self.state + directions[action]
        
        # Debug: Print current state and action
        print(f"Current state: {self.state.cpu().numpy()}, Action: {action}, New position: {new_position.cpu().numpy()}")

        # Check that new position is not a boundary or wall
        if (0 <= new_position[0] < self.grid_size and
            0 <= new_position[1] < self.grid_size):
            
            if self.grid[new_position[1], new_position[0]] != 1:
                self.state = new_position
                reward = 0.0
                done = False

                # Check for mandatory point
                if self.grid[new_position[1], new_position[0]] == 2:
                    self.visited_mandatory_pts.add((new_position[0].item(), new_position[1].item()))
                    reward = 1.0  # Reward for visiting a mandatory point
                    print(f"Visited Mandatory Point: ({new_position[0].item()}, {new_position[1].item()})")

                # Check if the goal is reached (top right corner)
                if th.equal(self.state, th.tensor([self.grid_size-1, self.grid_size-1], device=self.device)):
                    done = True

                # Pheromone update
                self.pheromones[self.state[0], self.state[1]] += 1.0  # Pheromone added to current position
                self.evaporate_pheromones()  # Pheromones evaporate over time

                print(f"Action taken: {action}, New state: {self.state}, Reward: {reward}, Done: {done}")
                return self.state.cpu().numpy(), reward, done, False, {}  # Add empty info dictionary
            else:
                # Penalize for hitting the wall
                reward = -0.5
                print(f"Hit wall. Action: {action}, State: {self.state.cpu().numpy()}, Reward: {reward}")
                return self.state.cpu().numpy(), reward, False, False, {}  # Add empty info dictionary
        else:
            # Penalize for hitting the boundary
            reward = -0.5
            print(f"Hit boundary. Action: {action}, State: {self.state.cpu().numpy()}, Reward: {reward}")
            return self.state.cpu().numpy(), reward, False, False, {}  # Add empty info dictionary 
        

    def evaporate_pheromones(self):
        evaporation_rate = 0.5
        self.pheromones *= (1.0-evaporation_rate) # pheromones evaporate over time
    
    def ant_move(self, tabu_length=2, max_steps=100):
        path = []
        tabu = []
        self.reset()
        goal_state = th.tensor([self.grid_size-1, self.grid_size-1], device=self.device, dtype=th.int32)
        step = 0
        while not th.equal(self.state, goal_state) or len(self.vistited_mandatory_pts) < (self.grid==2).sum().item():
            x, y = self.state.tolist()
            # print(f"Points visited: {len(self.vistited_mandatory_pts)}")
            # print(f"Total points: {(self.grid==2).sum().item()}") # debugging
            #print(f"Current state: ({x}, {y})")
            possible_actions = []
            if y < self.grid_size-1 and self.grid[y+1, x] == 0: # Move up
                possible_actions.append((x, y+1))
            if y > 0 and self.grid[y-1, x] == 0: # Move down
                possible_actions.append((x, y-1))
            if x > 0 and self.grid[y, x-1] == 0: # Mpve left
                possible_actions.append((x-1, y))
            if x < self.grid_size-1 and self.grid[y, x+1] == 0: # Move right
                possible_actions.append((x+1, y))
            
            if not possible_actions:
                break # No moves possible

            
            # Choosing the next move based on pheromones

            pheromone_levels = th.tensor([self.pheromones[ny, nx] for nx, ny in possible_actions], device=self.device)
            total_pheromone = pheromone_levels.sum()
            exploration_factor = th.rand(len(possible_actions), device=self.device)
            probs = (pheromone_levels + exploration_factor) / (total_pheromone + exploration_factor.sum())

            # Ensure probabilities are valid
            if th.isnan(probs).any() or th.isinf(probs).any() or (probs < 0).any():
                raise ValueError("Invalid probabilities: {}".format(probs))
            next_move_idx = th.multinomial(probs, 1).item()
            next_move = possible_actions[next_move_idx]
            if next_move in tabu:
                if len(possible_actions) == 1:
                    path.append(next_move)
                    tabu.remove(next_move)
                else:
                    continue
            path.append(next_move)
            tabu.append(next_move)

            if len(tabu) > tabu_length:
                tabu.pop(0)
            if self.grid[self.state[1], self.state[0]] == 2:
                self.visited_mandatory_points.add((self.state[0].item(), self.state[1].item()))
                print(f"Visited Mandatory Point: ({self.state[0].item()}, {self.state[1].item()})")  # Debugging: Print when a mandatory point is visited
            self.state = th.tensor(next_move, device=self.device, dtype=th.int32)  
            step +=1
            if step > max_steps:
                print("Max steps reached")
                break 

        return path
    def update_pheromones(self, paths):
        for path in paths:
            for x, y in path:
                self.pheromones[y, x] += 1.0 / len(path) # Depositing pheromones along the path


                if self.pheromones[y, x] > self.max_pheromone:
                    self.pheromones[y, x] = self.max_pheromone


    def run_aco(self, num_ants=5, num_iterations=1, evaporation_rate=0.1, pheromone_deposit=1.0, exploration_prob=0.1):
        best_path = None
        best_path_length = np.inf

        for i in range(num_iterations):
            paths = []
            for ant_index in range(num_ants):
                # if np.random.rand() < exploration_prob:
                #     path = self.random_path(self.state)
                # else:
                path = self.ant_move()
                paths.append(path)
                path_length = len(path)

                # Debug: Print the path taken by each ant
                print(f"Ant {ant_index + 1}/{num_ants} path: {path}, Path length: {path_length}")

                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

            self.update_pheromones(paths)
            print(f"Iteration {i+1}/{num_iterations} completed.")
            print(f"Best path found: {best_path} with length {best_path_length}")
        
        self.best_path = best_path
        
    # def random_path(self, start_position):
    #     path = []
    #     current_position = start_position
    #     goal_position = (self.grid_size - 1, self.grid_size - 1)  # Assuming the goal is at the bottom-right corner

    #     while current_position != goal_position:
    #         path.append(current_position)
    #         possible_moves = self.get_possible_moves(current_position)
    #         if not possible_moves:
    #             break  # No more valid moves, break the loop
    #         current_position = possible_moves[np.random.randint(len(possible_moves))]

    #     path.append(goal_position)  # Ensure the goal is included in the path
    #     return path

    # def get_possible_moves(self, position):
    #     x, y = position
    #     possible_moves = []

    #     # Check all four possible moves (up, down, left, right)
    #     if x > 0 and self.grid[y, x - 1] != 1:  # Move left
    #         possible_moves.append((x - 1, y))
    #     if x < self.grid_size - 1 and self.grid[y, x + 1] != 1:  # Move right
    #         possible_moves.append((x + 1, y))
    #     if y > 0 and self.grid[y - 1, x] != 1:  # Move up
    #         possible_moves.append((x, y - 1))
    #     if y < self.grid_size - 1 and self.grid[y + 1, x] != 1:  # Move down
    #         possible_moves.append((x, y + 1))

    #     return possible_moves

    def visualize_pheromones(self):
        # Transfer the tensor to the CPU and convert it to a NumPy array
        pheromones_cpu = self.pheromones.cpu().numpy()
        plt.imshow(pheromones_cpu, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Pheromone Levels")
        plt.show()

    def render(self):
        if self.render_mode == "human":
            # Create a grid of spaces using NumPy
            grid = np.full((self.grid_size, self.grid_size), ' ', dtype=str)
            
            # Populate the grid with walls
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.grid[i, j] == 1:
                        grid[i, j] = 'W'  # Representation of Wall
                    if self.grid[i, j] == 2:
                        grid[i, j] = 'M'  # Representation of Mandatory Point
            
            # Populate the grid with the best path if it exists
            if hasattr(self, 'best_path'):
                for step in self.best_path:
                    grid[step[1], step[0]] = 'O'  # Representation of Optimal Path
            
            # Mark the player's current position
            grid[self.state[1], self.state[0]] = 'P'  # Representation of Player
            
            # Mark the goal position
            grid[self.grid_size - 1, self.grid_size - 1] = 'G'  # Representation of Goal
            
            # Print the grid
            print("\n".join(["|".join(row) for row in grid]))