import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import itertools
import time
import random
from collections import deque
        
class GridEnv(gym.Env):
    def __init__(self, render_mode=None, device='cuda'):
        super(GridEnv, self).__init__()

        self.grid_size = 11  # grid size
        self.action_space = spaces.Discrete(4)  # 4 possible actions (up, down, left, right)
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)  # 2D position
        self.render_mode = render_mode
        self.device = device

        # Grid definition

        self.grid = th.tensor([
            [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 2, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 0, 1, 1, 2, 0, 1, 1, 0, 0], 
            [1, 2, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 1, 1, 2, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0],
            [1, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0]
        ], dtype=th.float32, device=self.device)

        self.pheromones = th.rand_like(self.grid, dtype=th.float32, device=self.device)*0.1 + 0.1# Initial pheromone level
        self.max_pheromone = 500.0  # Maximum pheromone level

        # Increase pheromone levels at mandatory points
        mandatory_points = (self.grid == 2).nonzero(as_tuple=True)
        self.pheromones[mandatory_points] = 4.0  # Set higher initial pheromone level at mandatory points
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = th.tensor([0, 0], device=self.device, dtype=th.int32)  # Start at bottom left corner
        self.pheromones = th.ones_like(self.grid, dtype=th.float32, device=self.device)*0.1
        self.visited_mandatory_pts = set()
        print("Environment reset. Initial state:", self.state.cpu().numpy())
        return self.state.cpu().numpy(), {}

    # def step(self, action):
    #     # Movement directions
    #     directions = th.tensor([
    #         [-1, 0],  # Up
    #         [1, 0],   # Down
    #         [0, -1],  # Left
    #         [0, 1]    # Right
    #     ], device=self.device, dtype=th.int32)

    #     # Calculation of new position based on previous action
    #     new_position = self.state + directions[action]
        
    #     # Debug: Print current state and action
    #     # print(f"Current state: {self.state.cpu().numpy()}, Action: {action}, New position: {new_position.cpu().numpy()}")

    #     # Check that new position is not a boundary or wall
    #     if (0 <= new_position[0] < self.grid_size and
    #         0 <= new_position[1] < self.grid_size):
            
    #         if self.grid[new_position[1], new_position[0]] != 1:
    #             self.state = new_position
    #             reward = 0.0
    #             done = False

    #             # Check for mandatory point
    #             if self.grid[new_position[1], new_position[0]] == 2:
    #                 self.visited_mandatory_pts.add((new_position[0].item(), new_position[1].item()))
    #                 reward = 1.0  # Reward for visiting a mandatory point
    #                 print(f"Visited Mandatory Point: ({new_position[0].item()}, {new_position[1].item()})")

    #             # Check if the goal is reached (top right corner)
    #             if th.equal(self.state, th.tensor([self.grid_size-1, self.grid_size-1], device=self.device)):
    #                 done = True

    #             # Pheromone update
    #             self.pheromones[self.state[0], self.state[1]] += 1.0  # Pheromone added to current position
    #             self.evaporate_pheromones()  # Pheromones evaporate over time

    #             print(f"Action taken: {action}, New state: {self.state}, Reward: {reward}, Done: {done}")
    #             return self.state.cpu().numpy(), reward, done, False, {}  # Add empty info dictionary
    #         else:
    #             # Penalize for hitting the wall
    #             reward = -0.5
    #             print(f"Hit wall. Action: {action}, State: {self.state.cpu().numpy()}, Reward: {reward}")
    #             return self.state.cpu().numpy(), reward, False, False, {}  # Add empty info dictionary
    #     else:
    #         # Penalize for hitting the boundary
    #         reward = -0.5
    #         print(f"Hit boundary. Action: {action}, State: {self.state.cpu().numpy()}, Reward: {reward}")
    #         return self.state.cpu().numpy(), reward, False, False, {}  # Add empty info dictionary 
        

    def evaporate_pheromones(self):
        evaporation_rate = 0.3
        self.pheromones *= (1.0-evaporation_rate) # pheromones evaporate over time
    
    def heuristic(self, x, y):
        goal_x, goal_y = self.grid_size - 1, self.grid_size - 1
        goal_distance = th.norm(th.tensor([goal_x - x, goal_y - y], device=self.device, dtype=th.float32))+1e-6

        # Distance to mandatory points
        mandatory_points = th.nonzero(self.grid == 2)
        if len(mandatory_points) > 0:
            mandatory_distances = th.norm(mandatory_points - th.tensor([x, y], device=self.device, dtype=th.float32), dim=1)
            min_mandatory_distance = mandatory_distances.min().item() + 1e-6
        else:
            min_mandatory_distance = 1e-6 # Avoid division by zero for no mandatory points

        # Combine the two heuristics

        combined_heuristic = 1.0 / (goal_distance + min_mandatory_distance)
        return combined_heuristic                 
    
    def ant_move(self, tabu_length=3, max_steps=1000):
        path = []
        tabu = [(0, 0)]
        self.reset()
        goal_state = th.tensor([self.grid_size-1, self.grid_size-1], device=self.device, dtype=th.int32)
        step = 0
        backtrack = False
        previous_position = None
        while not th.equal(self.state, goal_state) or len(self.visited_mandatory_pts) < (self.grid==2).sum().item():
            x, y = self.state.tolist()
            # print(f"Points visited: {len(self.visited_mandatory_pts)}")
            # print(f"Total points: {(self.grid==2).sum().item()}") # debugging
            # print(f"Current state: ({x}, {y})")
            possible_actions = []
            if y < self.grid_size-1 and self.grid[y+1, x] != 1: # Move up
                possible_actions.append((x, y+1))
            if y > 0 and self.grid[y-1, x] != 1: # Move down
                possible_actions.append((x, y-1))
            if x > 0 and self.grid[y, x-1] != 1: # Move left
                possible_actions.append((x-1, y))
            if x < self.grid_size-1 and self.grid[y, x+1] != 1: # Move right
                possible_actions.append((x+1, y))
            
            # Ensure it is not a tabu move
            possible_actions = [action for action in possible_actions if action not in tabu]

            if not possible_actions:
                print("No moves possible. Breaking.")
                return path, False # No moves possible
            #print(f'Backtrack: {backtrack}')            
            if backtrack:
                next_move = previous_position
            else:

                # Debug: Print possible actions
                # print(f"Possible actions: {possible_actions}")

                # Choosing the next move based on pheromones

                pheromone_levels = th.tensor([self.pheromones[ny, nx] for nx, ny in possible_actions], device=self.device)
                total_pheromone = pheromone_levels.sum()
                exploration_factor = th.rand(len(possible_actions), device=self.device)
                heuristic = th.tensor([self.heuristic(nx, ny) for nx, ny in possible_actions], device=self.device)
                probs = (pheromone_levels + exploration_factor + heuristic) / (total_pheromone + exploration_factor.sum() + heuristic.sum())

                # Ensure probabilities are valid
                if th.isnan(probs).any() or th.isinf(probs).any() or (probs < 0).any():
                    raise ValueError("Invalid probabilities: {}".format(probs))
                next_move_idx = th.multinomial(probs, 1).item()
                next_move = possible_actions[next_move_idx]
            
            if self.grid[self.state[1], self.state[0]] == 2:
                mandatory_pt = (self.state[0].item(), self.state[1].item())
                if mandatory_pt not in self.visited_mandatory_pts:
                    self.visited_mandatory_pts.add(mandatory_pt)
                    # print(f"Visited Mandatory Point: {mandatory_pt})")
                    tabu.clear()
                    tabu.append(mandatory_pt)
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
            
            # print(f"Next move: {next_move}")
            # print(f"Mandatory points visited: {self.visited_mandatory_pts}")
            self.state = th.tensor(next_move, device=self.device, dtype=th.int32)  
            step +=1
            if step > max_steps:
                print("Max steps reached")
                return path, False
            # if th.equal(self.state, goal_state) and len(self.visited_mandatory_pts) == (self.grid == 2).sum().item():
            #     print("Goal reached and all mandatory points visited.")
            # else:
            #     print("Failed to reach goal or visit all mandatory points.")

        return path, True
    
    def is_valid_path(self, path):
        visited_mandatory_pts = set()
        for x, y in path:
            if self.grid[y, x] == 2:
                visited_mandatory_pts.add((x, y))
        goal_state = (self.grid_size-1, self.grid_size-1)
        return (goal_state in path) and (len(visited_mandatory_pts) == (self.grid == 2).sum().item())
    
    def distance(self, p1, p2):
        return th.norm(th.tensor(p1, dtype=th.float32) - th.tensor(p2, dtype=th.float32))
    
    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.distance(path[i], path[i + 1])
        return length

    def three_opt(self, path):
        best_path = path
        best_path_length = self.calculate_path_length(path)
        start_time = time.time()
        for i in range(len(path) - 2):
            for j in range(i + 1, min(i + 10, len(path) - 1)):
                for k in range(max(j + 1, len(path) - 10), len(path)):
                    new_paths = self.three_opt_swap(path, i, j, k)
                    for new_path in new_paths:  
                        new_path_length = self.calculate_path_length(new_path)
                        if new_path_length < best_path_length:
                            best_path = new_path
                            best_path_length = new_path_length
                            elapsed_time = time.time() - start_time
                            print(f"New best path length: {best_path_length} in {elapsed_time:.2f} seconds ")
        return best_path



    def three_opt_swap(self, path, i, j, k):
       
        new_paths = []
        new_paths.append(path[:i] + path[i:j][::-1] + path[j:k][::-1] + path[k:])
        new_paths.append(path[:i] + path[j:k] + path[i:j] + path[k:])
        new_paths.append(path[:i] + path[j:k][::-1] + path[i:j] + path[k:])
        new_paths.append(path[:i] + path[j:k] + path[i:j][::-1] + path[k:])
        new_paths.append(path[:i] + path[i:j][::-1] + path[j:k] + path[k:])

        return new_paths

    

    def update_pheromones(self, paths, evaporation_rate, pheromone_deposit):
        self.pheromones *= (1.0 - evaporation_rate)  # Pheromones evaporate over time
        best_path = min(paths, key=lambda path: self.calculate_path_length(path))
        for path in paths:
            for (x, y) in path:
                if self.pheromones[y, x] < self.max_pheromone:
                    self.pheromones[y, x] += pheromone_deposit
            for (x, y) in best_path:
                if self.pheromones[y, x] < self.max_pheromone:
                    self.pheromones[y, x] += pheromone_deposit * 4.0 # Elitist strategy
            
            
    def get_mandatory_points(self):
        mandatory_points = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] == 2:
                    mandatory_points.append((i, j))
        return mandatory_points

    def run_aco(self, num_ants=30, num_iterations=8, evaporation_rate=0.8, pheromone_deposit=0.5):
        best_path = None
        best_path_length = np.inf
        mandatory_pts = self.get_mandatory_points()

        best_path_lengths = []
        ant_steps_per_iteration = []
        for i in range(num_iterations):
            paths = []
            steps_per_ant = []
            for ant_index in range(num_ants):
                # if np.random.rand() < exploration_prob:
                #     path = self.random_path(self.state, mandatory_pts)
                # else:
                path, success = self.ant_move()
                if success:
                    paths.append(path)
                    path_length = len(path)
                    steps_per_ant.append(path_length)

                # Debug: Print the path taken by each ant
                #print(f"Ant {ant_index + 1}/{num_ants} path: {path}, Path length: {path_length}")

                    if path_length < best_path_length:
                        best_path = path
                        best_path_length = path_length


            self.update_pheromones(paths, evaporation_rate, pheromone_deposit)
            print(f"Iteration {i+1}/{num_iterations} completed.")
            print(f"Best path found: {best_path} with length {best_path_length}")
            best_path_lengths.append(best_path_length)
            ant_steps_per_iteration.append(steps_per_ant)
        optimized_path = self.local_search(best_path) # Apply local 3-opt search  
        optimized_path_length = self.calculate_path_length(optimized_path)  
        print(f"Best path found: {optimized_path} with length {optimized_path_length}")
        
        self.best_path = optimized_path

        #Plot data
        self.plot_ant_steps_per_iteration(ant_steps_per_iteration)
        self.plot_best_path_lengths(best_path_lengths)
        self.plot_best_path()

    def local_search(self, path):
        return self.three_opt(path)

    def plot_best_path_lengths(self, best_path_lengths):
        plt.figure()
        plt.plot(best_path_lengths, marker='o')
        plt.title('Evolution of Best Path Length Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Path Length')
        plt.grid(True)
        plt.show()

    def plot_ant_steps_per_iteration(self, ant_steps_per_iteration):
        plt.figure()
        for i, steps in enumerate(ant_steps_per_iteration):
            plt.plot(steps, marker='o', label=f'Iteration {i+1}')
        plt.title('Number of Steps Each Ant Takes Within Each Iteration')
        plt.xlabel('Ant Index')
        plt.ylabel('Number of Steps')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_best_path(self):
        grid = self.grid.cpu().numpy()
        best_path = self.best_path

        plt.figure(figsize=(10, 10))
        plt.imshow(grid, cmap='gray_r', origin='upper')

        # Overlay the best path
        if best_path:
            path_x, path_y = zip(*best_path)
            plt.plot(path_x, path_y, marker='o', color='red', label='Best Path')

        # Mark start and goal
        plt.scatter(0, 0, color='green', s=100, label='Start')
        plt.scatter(self.grid_size-1, self.grid_size-1, color='blue', s=100, label='Goal')

        plt.title('Best Path in Grid Environment')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()
            
    # def random_path(self, start_position, mandatory_pts=[]):
    #     path = []
    #     current_position = start_position
    #     goal_position = (self.grid_size - 1, self.grid_size - 1)  # Assuming the goal is at the bottom-right corner
    #     pts_to_visit = mandatory_pts + [goal_position]
    #     steps = 0
    #     max_steps = 1000
    #     print('Random path being used')
    #     while pts_to_visit:
    #         next_points = pts_to_visit.pop()
    #         while current_position != next_points:
    #             path.append(current_position)
    #             possible_moves = self.get_possible_moves(current_position)
    #             if not possible_moves:
    #                 break  # No more valid moves, break the loop
    #             current_position = possible_moves[np.random.randint(len(possible_moves))]
    #             print(f"Step: {steps}, Current position: {current_position}")
    #             steps += 1
    #             if steps > max_steps:
    #                 print("Max steps reached.")
    #                 break
    #         path.append(next_points)  # Ensure the goal is included in the path
    #         current_position = next_points
            
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