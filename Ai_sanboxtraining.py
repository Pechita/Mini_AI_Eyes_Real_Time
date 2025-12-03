import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union

# --- 1. World / Units Configuration (Constants) ---
FEET_TO_PX = 50        # 1 ft = 50 px
WORLD_FT = 15.0
WORLD_PX = int(WORLD_FT * FEET_TO_PX) # 750 px

# --- EXPLORATION & GRID CONFIGURATION ---
GRID_CELL_SIZE_PX = 25.0 # 30x30 Grid
WORLD_GRID_W = int(WORLD_PX // GRID_CELL_SIZE_PX) 
WORLD_GRID_H = int(WORLD_PX // GRID_CELL_SIZE_PX) 

NUM_SAMPLES = 72       # Lidar rays (5 degree resolution)
DEG_PER_SAMPLE = 360.0 / NUM_SAMPLES 
ROBOT_RADIUS_PX = 10 
MOVE_SAFETY_BUFFER_PX = 15 
FIXED_JUMP_DISTANCE_PX = 50.0 

# --- REINFORCEMENT LEARNING CONFIGURATION (Initial/Fixed Values) ---
BASE_RL_CONFIG = {
    "TEST_EPISODES": 10, 
    "LEARNING_RATE": 0.1,    
    "DISCOUNT_FACTOR": 0.9,  
    "EXPLORATION_RATE": 1.0, # Initial Epsilon
    "MIN_EXPLORATION_RATE": 0.01,
    "ACTION_SPACE_SIZE": 8,  
    "STATE_BIN_COUNT": 3,    
    "MAX_STEPS_PER_EPISODE": 200,
}

# --- REWARD SHAPING ---
REWARD_SHAPING = {
    "INFO_GAIN_MULTIPLIER": 5000.0, 
    "MOVE_COST": -1.0,              
    "STUCK_PENALTY": -50.0,         
    "GOAL_BONUS": 500.0,            
}

# Global dictionary to store runtime configuration
RL_CONFIG = BASE_RL_CONFIG.copy()
TARGET_COVERAGE_SCORE = 0.95 

# --- 2. SHAPES CONFIG (Environment Geometry) ---
SHAPES_CONFIG = {
    "rectangles": [
        {"cx_ft": 7.5, "cy_ft": 10.0, "w_ft": 3.0, "h_ft": 1.5},
        {"cx_ft": 5.0, "cy_ft": 6.0, "w_ft": 2.0, "h_ft": 2.0},
    ],
    "circles": [
        {"cx_ft": 10.0, "cy_ft": 5.0, "r_ft": 1.0},
    ],
}

# --- 3. Geometry Primitives & World ---
Point = Tuple[float, float]
Segment = Tuple[Point, Point]

@dataclass
class Rectangle:
    cx: float; cy: float; w: float; h: float
    def segments(self) -> List[Segment]:
        x0 = self.cx - self.w/2.0; x1 = self.cx + self.w/2.0
        y0 = self.cy - self.h/2.0; y1 = self.cy + self.h/2.0
        return [((x0,y0),(x1,y0)), ((x1,y0),(x1,y1)), ((x1,y1),(x0,y1)), ((x0,y1),(x0,y0))] 

@dataclass
class Circle:
    cx: float; cy: float; r: float

@dataclass
class World:
    size_px: int = WORLD_PX
    lidar_pos: Point = field(default_factory=lambda: (WORLD_PX/4, WORLD_PX/2))
    lidar_heading_deg: float = field(default_factory=lambda: 0.0)
    rectangles: List[Rectangle] = field(default_factory=list)
    circles: List[Circle] = field(default_factory=list)

    def wall_segments(self) -> List[Segment]:
        s = self.size_px
        return [((0,0),(s,0)), ((s,0),(s,s)), ((s,s),(0,s)), ((0,s),(0,0))]

    def all_segments(self) -> List[Segment]:
        segs = list(self.wall_segments())
        for r in self.rectangles:
            segs.extend(r.segments())
        return segs

def ray_segment_intersection(ray_o: Point, ray_d: Point, p0: Point, p1: Point) -> Optional[float]:
    """ Finds the intersection parameter 't' where ray hits segment. """
    ox, oy = ray_o; dx, dy = ray_d
    x1, y1 = p0; x2, y2 = p1
    vx, vy = x2 - x1, y2 - y1
    denom = dx * (-vy) + dy * (vx)
    if abs(denom) < 1e-9: return None 
    rx, ry = x1 - ox, y1 - oy
    t = (rx * (-vy) + ry * (vx)) / denom
    if t < 0: return None
    ix, iy = ox + t*dx, oy + t*dy
    segment_len_sq = vx*vx + vy*vy
    if segment_len_sq < 1e-9: return None 
    dot_product = (ix - x1)*(x2 - x1) + (iy - y1)*(y2 - y1)
    u = dot_product / segment_len_sq
    if u < -1e-9 or u > 1.0 + 1e-9: return None
    return t

def ray_circle_intersection(ray_o: Point, ray_d: Point, center: Point, r: float) -> Optional[float]:
    """ Finds the intersection parameter 't' where ray hits circle. """
    ox, oy = ray_o; dx, dy = ray_d
    cx, cy = center
    fx, fy = ox - cx, oy - cy 
    a = dx*dx + dy*dy
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c
    if disc < 0: return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    ts = [t for t in (t1, t2) if t >= 0]
    if not ts: return None
    return min(ts)

def simulate_frame(world: World, num_samples: int = NUM_SAMPLES, range_px: float = WORLD_PX * 2) -> List[float]:
    """ Simulates Lidar readings and returns distances in PX. """
    lx, ly = world.lidar_pos
    distances_px: List[float] = [range_px]*num_samples
    segs = world.all_segments()
    
    for i in range(num_samples):
        angle_deg = world.lidar_heading_deg - i*DEG_PER_SAMPLE
        th = math.radians(angle_deg)
        dx, dy = math.cos(th), math.sin(th)
        best_t: Optional[float] = None
        
        for p0, p1 in segs:
            t = ray_segment_intersection((lx,ly), (dx,dy), p0, p1)
            if t is not None and (best_t is None or t < best_t): best_t = t
        
        for c in world.circles:
            t = ray_circle_intersection((lx,ly), (dx,dy), (c.cx,c.cy), c.r)
            if t is not None and (best_t is None or t < best_t): best_t = t
            
        if best_t is None or best_t > range_px: best_t = range_px
        distances_px[i] = best_t
        
    return distances_px

# --- 4. Map and Agent Logic (RL Environment) ---

class MapGrid:
    def __init__(self, world: World):
        self.W = WORLD_GRID_W
        self.H = WORLD_GRID_H
        self.cell_size = GRID_CELL_SIZE_PX
        self.visited_grid = np.zeros((self.W, self.H), dtype=np.int8) 
        self.ground_truth = self._generate_ground_truth(world) 
        self.total_mappable_cells = np.sum(self.ground_truth == 0)

    def _px_to_grid(self, x_px: float, y_px: float) -> Optional[Tuple[int, int]]:
        """ Converts pixel coordinates to grid cell indices (i, j). """
        i = int(x_px // self.cell_size); j = int(y_px // self.cell_size)
        if 0 <= i < self.W and 0 <= j < self.H: return i, j
        return None
        
    def _generate_ground_truth(self, world: World) -> np.ndarray:
        """ Pre-calculates fixed obstacle cells (2) and free cells (0). """
        gt = np.zeros((self.W, self.H), dtype=np.int8)
        
        for i in range(self.W):
            for j in range(self.H):
                cx = (i + 0.5) * self.cell_size
                cy = (j + 0.5) * self.cell_size
                
                for r in world.rectangles: 
                    if r.cx - r.w/2 < cx < r.cx + r.w/2 and r.cy - r.h/2 < cy < r.cy + r.h/2:
                        gt[i, j] = 2; break
                if gt[i,j] == 2: continue

                for c in world.circles: 
                    if math.hypot(cx - c.cx, cy - c.cy) < c.r:
                        gt[i, j] = 2; break
                if gt[i,j] == 2: continue
                        
        return gt

    def update_map(self, robot_pos: Point, distances_px: List[float]):
        """ Updates the occupancy grid based on the current Lidar scan. """
        lx, ly = robot_pos
        coords = self._px_to_grid(lx, ly)
        if coords and self.ground_truth[coords] == 0:
             self.visited_grid[coords] = 1

        for i, dist in enumerate(distances_px):
            angle_deg = -i*DEG_PER_SAMPLE
            th = math.radians(angle_deg)
            dx, dy = math.cos(th), math.sin(th)
            
            # Mark Free Space
            for d in np.arange(0, dist - self.cell_size, self.cell_size):
                fx = lx + dx * d; fy = ly + dy * d
                f_coords = self._px_to_grid(fx, fy)
                if f_coords and self.ground_truth[f_coords] == 0:
                    self.visited_grid[f_coords] = 1
            
            # Mark Occupied Space
            hit_x = lx + dx * dist; hit_y = ly + dy * dist
            h_coords = self._px_to_grid(hit_x, hit_y)
            if h_coords and self.ground_truth[h_coords] == 2:
                self.visited_grid[h_coords] = 2

    def calculate_coverage(self) -> float:
        """ Returns the ratio of mapped free cells to total mappable free cells. """
        current_mapped_free_cells = np.sum(np.logical_and(self.visited_grid == 1, self.ground_truth == 0))
        return current_mapped_free_cells / self.total_mappable_cells if self.total_mappable_cells > 0 else 0.0

    def reset(self):
        """ Resets the visited map for a new episode. """
        self.visited_grid.fill(0)


class RLExplorer:
    def __init__(self, initial_world: World, map_grid: MapGrid):
        self.initial_world = initial_world
        self.world = initial_world
        self.map_grid = map_grid
        
        # RL Parameters
        self.q_table: Dict[Tuple, np.ndarray] = {} 
        self.alpha = RL_CONFIG["LEARNING_RATE"]
        self.gamma = RL_CONFIG["DISCOUNT_FACTOR"]
        self.epsilon = RL_CONFIG["EXPLORATION_RATE"]
        self.decay_rate = RL_CONFIG["DECAY_RATE"]
        
        self.current_state: Optional[Tuple] = None

    def reset_episode(self, start_pos: Point):
        """ Resets the agent and environment for a new training episode. """
        # Create a new world instance for the episode
        self.world = load_initial_world() 
        self.world.lidar_pos = start_pos
        self.map_grid.reset()
        
        initial_scan = simulate_frame(self.world)
        self.map_grid.update_map(self.world.lidar_pos, initial_scan)
        self.current_state = self._get_discretized_lidar_state(initial_scan)

    def _get_discretized_lidar_state(self, scan_results: List[float]) -> Tuple[int, ...]:
        """ Converts the 72 Lidar readings into a compact, discrete state tuple (8 bins, 3 categories). """
        distances = np.array(scan_results)
        bins = np.array_split(distances, RL_CONFIG["ACTION_SPACE_SIZE"])
        
        state_tuple = []
        max_range = WORLD_PX * 2 
        
        # Distance boundaries for 3 bins
        bin_bounds = [max_range / RL_CONFIG["STATE_BIN_COUNT"], 2 * max_range / RL_CONFIG["STATE_BIN_COUNT"]]

        for b in bins:
            avg_dist = np.median(b) 
            
            if avg_dist < bin_bounds[0]: state_tuple.append(0) # Near
            elif avg_dist < bin_bounds[1]: state_tuple.append(1) # Mid
            else: state_tuple.append(2) # Far
                
        return tuple(state_tuple)

    def _get_action(self, state: Tuple[int, ...]) -> int:
        """ Epsilon-greedy action selection. """
        q_values = self.q_table.get(state, np.zeros(RL_CONFIG["ACTION_SPACE_SIZE"]))
        
        if random.random() < self.epsilon:
            return random.randrange(RL_CONFIG["ACTION_SPACE_SIZE"]) # Explore 
        else:
            return np.argmax(q_values) # Exploit

    def _update_q_table(self, state: Tuple, action: int, reward: float, new_state: Tuple):
        """ Implements the Q-Learning update rule. """
        
        current_q_array = self.q_table.get(state, np.zeros(RL_CONFIG["ACTION_SPACE_SIZE"]))
        
        if new_state is None:
            max_future_q = 0.0
        else:
            future_q_array = self.q_table.get(new_state, np.zeros(RL_CONFIG["ACTION_SPACE_SIZE"]))
            # Q-Learning uses the maximum Q-value of the next state
            max_future_q = np.max(future_q_array)
        
        old_q = current_q_array[action]
        
        # Q(s, a) = (1 - a) * Q(s, a) + a * (R + y * max(Q(s', a')))
        new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * max_future_q)
        
        current_q_array[action] = new_q
        self.q_table[state] = current_q_array

    def step(self, action: int) -> Tuple[Optional[Tuple], float, bool, Dict]:
        """ Takes an action, moves the agent, and returns (new_state, reward, done, info). """
        
        angle_deg = action * 360.0 / RL_CONFIG["ACTION_SPACE_SIZE"]
        th = math.radians(angle_deg)
        
        # Determine maximum safe jump distance
        current_scan = simulate_frame(self.world)
        action_bin_start = action * (NUM_SAMPLES // RL_CONFIG["ACTION_SPACE_SIZE"])
        dist_in_sector = np.min(current_scan[action_bin_start : action_bin_start + (NUM_SAMPLES // RL_CONFIG["ACTION_SPACE_SIZE"])])

        move_distance_px = min(FIXED_JUMP_DISTANCE_PX, max(0.0, dist_in_sector - MOVE_SAFETY_BUFFER_PX))

        if move_distance_px < self.map_grid.cell_size:
            reward = REWARD_SHAPING["STUCK_PENALTY"]
            done = False 
            new_state = self.current_state 
            return new_state, reward, done, {"moved": False, "coverage": self.map_grid.calculate_coverage()}
        
        # Execute Move
        lx, ly = self.world.lidar_pos
        new_x = lx + math.cos(th) * move_distance_px
        new_y = ly + math.sin(th) * move_distance_px
        
        self.world.lidar_pos = (new_x, new_y)
        self.world.lidar_heading_deg = angle_deg
        
        # Calculate Reward and Update Map
        initial_coverage = self.map_grid.calculate_coverage()
        
        new_scan_results = simulate_frame(self.world)
        self.map_grid.update_map(self.world.lidar_pos, new_scan_results)
        
        final_coverage = self.map_grid.calculate_coverage()
        info_gain = final_coverage - initial_coverage
        
        reward = (
            info_gain * REWARD_SHAPING["INFO_GAIN_MULTIPLIER"] + 
            REWARD_SHAPING["MOVE_COST"] # Penalty for taking a turn/step
        )
        
        # Check for Termination
        done = final_coverage >= TARGET_COVERAGE_SCORE
        
        if done:
            reward += REWARD_SHAPING["GOAL_BONUS"] 
        
        new_state = self._get_discretized_lidar_state(new_scan_results)
        
        return new_state, reward, done, {"moved": True, "coverage": final_coverage}

def load_initial_world() -> World:
    """ Helper to initialize the world based on static configuration. """
    world = World()
    for r in SHAPES_CONFIG.get("rectangles", []):
        cx = r["cx_ft"] * FEET_TO_PX; cy = r["cy_ft"] * FEET_TO_PX
        w  = r["w_ft"]  * FEET_TO_PX; h  = r["h_ft"]  * FEET_TO_PX
        world.rectangles.append(Rectangle(cx, cy, w, h))
    for c in SHAPES_CONFIG.get("circles", []):
        cx = c["cx_ft"] * FEET_TO_PX; cy = c["cy_ft"] * FEET_TO_PX
        rr = c["r_ft"]  * FEET_TO_PX
        world.circles.append(Circle(cx, cy, rr))
    return world

def plot_training_results(steps_history: List[int], num_episodes: int, avg_percentage: float = 0.05):
    """ 
    Generates a Matplotlib plot with two subplots: 
    1. Raw Steps per Episode (showing volatility).
    2. Adaptive Moving Average (showing trend). 
    """
    
    # Calculate adaptive window size
    window_size = int(num_episodes * avg_percentage)
    if window_size < 2:
        window_size = min(num_episodes, 2) # Ensure a minimum reasonable size
        
    print(f"\n--- Plotting Results ---")
    print(f"Using {avg_percentage*100:.0f}% Moving Average: Window Size = {window_size} episodes")

    
    if len(steps_history) < window_size:
        print("\nPlotting skipped: Not enough data points for a moving average.")
        return
        
    # Calculate moving average
    moving_avg = np.convolve(steps_history, np.ones(window_size)/window_size, mode='valid')
    episodes = np.arange(1, num_episodes + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Q-Learning Agent Performance: Steps Required to Achieve {TARGET_COVERAGE_SCORE*100:.0f}% Coverage', fontsize=16)
    
    # --- Subplot 1: Raw Data ---
    ax1.plot(episodes, steps_history, label='Steps per Episode (Raw)', alpha=0.6, color='skyblue')
    ax1.set_title('Raw Steps per Episode (Exploration Noise)', fontsize=12)
    ax1.set_ylabel('Steps (Turns) to Goal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # --- Subplot 2: Moving Average Trend ---
    # Adjust x-axis for the moving average (it starts 'window_size' episodes later)
    ma_episodes = np.arange(window_size, num_episodes + 1)
    ax2.plot(ma_episodes, moving_avg, label=f'{window_size}-Episode Moving Average ({avg_percentage*100:.0f}%)', color='darkorange', linewidth=2)
    ax2.set_title('Smoothed Learning Trend (Exploitation)', fontsize=12)
    ax2.set_xlabel('Training Episode')
    ax2.set_ylabel('Average Steps to Goal')
    
    # Add a horizontal line at the stable minimum (e.g., 12 steps from previous runs) for reference
    if len(moving_avg) > 0 and np.min(moving_avg) < 50:
        ax2.axhline(y=np.min(moving_avg) + 2, color='r', linestyle='--', alpha=0.7, label=f'Near Optimal Level (~{np.min(moving_avg)+2:.1f} Steps)')

    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
    plt.show()

def get_runtime_config():
    """ Prompts the user for runtime parameters and calculates the adaptive decay rate. """
    global RL_CONFIG, TARGET_COVERAGE_SCORE
    
    print("\n--- Setup Q-Learning Parameters ---")
    
    # 1. Get Target Coverage (F1 Threshold)
    while True:
        try:
            target = input(f"Enter target coverage threshold (e.g., 0.95 for 95%): ")
            target_coverage = float(target)
            if 0.0 < target_coverage <= 1.0:
                TARGET_COVERAGE_SCORE = target_coverage
                break
            else:
                print("Threshold must be between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # 2. Get Number of Episodes
    while True:
        try:
            num = input(f"Enter the number of training episodes (e.g., 500): ")
            num_episodes = int(num)
            if num_episodes > 1: # Must be > 1 to calculate decay
                RL_CONFIG["NUM_EPISODES"] = num_episodes
                break
            else:
                print("Number of episodes must be greater than 1.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # 3. Calculate Adaptive Decay Rate
    initial_epsilon = RL_CONFIG["EXPLORATION_RATE"]
    min_epsilon = RL_CONFIG["MIN_EXPLORATION_RATE"]
    num_episodes = RL_CONFIG["NUM_EPISODES"]
    
    # Decay Rate = (min_epsilon / initial_epsilon) ** (1 / num_episodes)
    decay_rate = (min_epsilon / initial_epsilon) ** (1 / num_episodes)
    RL_CONFIG["DECAY_RATE"] = decay_rate
    
    print(f"\nConfiguration set:")
    print(f"  Target Coverage: {TARGET_COVERAGE_SCORE*100:.1f}%")
    print(f"  Training Episodes: {num_episodes}")
    print(f"  Calculated Adaptive Epsilon Decay Rate: {decay_rate:.6f}")
    print("-----------------------------------")


# --- Main RL Training Execution ---

if __name__ == "__main__":
    
    get_runtime_config()
    
    initial_world = load_initial_world()
    map_grid = MapGrid(initial_world)
    explorer = RLExplorer(initial_world, map_grid)
    
    # Store metrics
    coverage_history: List[float] = []
    steps_history: List[int] = []
    
    start_pos = (WORLD_PX/4, WORLD_PX/2)
    num_episodes = RL_CONFIG["NUM_EPISODES"]

    print(f"--- Starting Q-Learning Training ---")
    print(f"Goal: {TARGET_COVERAGE_SCORE*100:.0f}% Coverage | State Size: 3^{RL_CONFIG['ACTION_SPACE_SIZE']} possible states")

    # =========================================================================
    # TRAINING PHASE
    # =========================================================================
    for episode in range(1, num_episodes + 1):
        
        explorer.reset_episode(start_pos)
        total_episode_reward = 0
        step_count = 0
        done = False
        current_state = explorer.current_state

        while not done and step_count < RL_CONFIG["MAX_STEPS_PER_EPISODE"]:
            
            action = explorer._get_action(current_state)
            new_state, reward, done, info = explorer.step(action)
            explorer._update_q_table(current_state, action, reward, new_state)
            
            current_state = new_state
            total_episode_reward += reward
            step_count += 1
            
            # Print status update in place
            print(f"TRAIN Ep {episode}/{num_episodes} | Step {step_count}/{RL_CONFIG['MAX_STEPS_PER_EPISODE']} | Coverage: {info['coverage']*100:.1f}% | Epsilon: {explorer.epsilon:.3f}", end="\r")

        # Decay epsilon and log episode results
        explorer.epsilon = max(RL_CONFIG["MIN_EXPLORATION_RATE"], explorer.epsilon * RL_CONFIG["DECAY_RATE"])
        final_coverage = explorer.map_grid.calculate_coverage()
        coverage_history.append(final_coverage)
        steps_history.append(step_count)

        # Print final episode result on a new line
        print(f"TRAIN Ep {episode:3d} | Steps: {step_count:3d} | Final Coverage: {final_coverage*100:.2f}% | Total Reward: {total_episode_reward:7.1f} | Epsilon: {explorer.epsilon:.3f}")

    # =========================================================================
    # EVALUATION PHASE
    # =========================================================================
    
    # Store test metrics
    test_steps: List[int] = []
    test_coverage: List[float] = []
    
    print("\n-----------------------------------------------------")
    print("Training Complete. Starting Evaluation Phase...")
    
    # Disable exploration (epsilon = 0) to use the learned policy purely
    explorer.epsilon = 0.0
    
    for episode in range(1, RL_CONFIG["TEST_EPISODES"] + 1):
        explorer.reset_episode(start_pos)
        step_count = 0
        done = False
        current_state = explorer.current_state

        while not done and step_count < RL_CONFIG["MAX_STEPS_PER_EPISODE"]:
            
            # Use only the learned policy (exploit)
            action = explorer._get_action(current_state)
            new_state, _, done, info = explorer.step(action)
            current_state = new_state
            step_count += 1
            
        final_coverage = explorer.map_grid.calculate_coverage()
        test_steps.append(step_count)
        test_coverage.append(final_coverage)
        
        print(f"EVAL Ep {episode:2d} | Steps: {step_count:3d} | Final Coverage: {final_coverage*100:.2f}%")

    # --- Final Test Results ---
    avg_test_steps = np.mean(test_steps)
    avg_test_coverage = np.mean(test_coverage)

    print("\n-----------------------------------------------------")
    print(f"EVALUATION SUMMARY ({RL_CONFIG['TEST_EPISODES']} episodes)")
    print(f"Policy based on {len(explorer.q_table)} learned states.")
    print(f"Average Steps to Achieve Coverage: {avg_test_steps:.1f} turns")
    print(f"Average Final Coverage: {avg_test_coverage*100:.2f}%")
    print("-----------------------------------------------------")
    
    # --- Generate Visual Plot ---
    try:
        plot_training_results(steps_history, num_episodes, avg_percentage=0.05) # Use 5% by default
    except Exception as e:
        # In case the environment lacks a display backend for matplotlib
        print(f"\n[ERROR] Could not generate Matplotlib plot. Please ensure you have 'matplotlib' installed and run this script in your local VS Code environment to view the graph. Details: {e}")