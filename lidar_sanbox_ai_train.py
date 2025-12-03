#This code demonstrates a simulation of lidar mapper and how it mapps.


import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import ttk

# 1. World / Units Configuration

FT_TO_MM = 304.8
FEET_TO_PX = 50  # 1 ft = 50 px
WORLD_FT = 15.0
WORLD_PX = int(WORLD_FT * FEET_TO_PX)  # 750 px

NUM_SAMPLES = 240
DEG_PER_SAMPLE = 360.0 / NUM_SAMPLES  # 1.5°

# Serial framing
START_TOKEN = "^"
END_TOKEN   = "$"

# 2. SHAPES CONFIG (The Environment)

SHAPES_CONFIG = {
    "rectangles": [
        {"cx_ft": 7.5, "cy_ft": 10.0, "w_ft": 3.0, "h_ft": 1.5},
        {"cx_ft": 5.0,  "cy_ft": 6.0,  "w_ft": 2.0, "h_ft": 2.0},
    ],
    "circles": [
        {"cx_ft": 10.0, "cy_ft": 5.0, "r_ft": 1.0},
        {"cx_ft": 2.0, "cy_ft": 5.0, "r_ft": 0.50},
        {"cx_ft": 0.0, "cy_ft": 10.0, "r_ft": 0.50},
    ],
}

# 3. Geometry Primitives & World

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
    lidar_pos: Point = field(default_factory=lambda: (WORLD_PX/2, WORLD_PX/2))
    lidar_heading_deg: float = -90.0
    rectangles: List[Rectangle] = field(default_factory=list)
    circles: List[Circle] = field(default_factory=list)

    def wall_segments(self) -> List[Segment]:
        s = self.size_px
        # Wall segments define the boundary of the world
        return [((0,0),(s,0)), ((s,0),(s,s)), ((s,s),(0,s)), ((0,s),(0,0))]

    def all_segments(self) -> List[Segment]:
        # Combines wall segments and all rectangle segments
        segs = list(self.wall_segments())
        for r in self.rectangles:
            segs.extend(r.segments())
        return segs

# 4. Ray Casting & Simulation

def ray_segment_intersection(ray_o: Point, ray_d: Point, p0: Point, p1: Point) -> Optional[float]:
    # Intersection logic for ray vs. line segment (unchanged)
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
    u = ((ix - x1)*vx + (iy - y1)*vy) / segment_len_sq
    if u < 0 or u > 1: return None
    return t

def ray_circle_intersection(ray_o: Point, ray_d: Point, center: Point, r: float) -> Optional[float]:
    # Intersection logic for ray vs. circle (unchanged)
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

def simulate_frame(world: World) -> List[int]:
    """Return 240 distances in millimeters."""
    lx, ly = world.lidar_pos
    distances_px: List[float] = [0.0]*NUM_SAMPLES
    segs = world.all_segments()

    for i in range(NUM_SAMPLES):
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

        if best_t is None: best_t = WORLD_PX * 2.0
        distances_px[i] = best_t

    px_to_mm = (1.0 / FEET_TO_PX) * FT_TO_MM
    distances_mm = [int(round(d * px_to_mm)) for d in distances_px]
    return distances_mm


# 5. Mapper Class (Optimized SLAM Core) - MODIFIED FOR SCORING AND COLLISION

class Mapper:
    """Consumes LiDAR data, updates the map (Occupancy Grid Set), and controls movement."""
    
    # Grid configuration (shared across the Mapper)
    GRID_CELL_SIZE_PX = 12.5 
    WORLD_GRID_W = int(WORLD_PX // GRID_CELL_SIZE_PX)
    WORLD_GRID_H = int(WORLD_PX // GRID_CELL_SIZE_PX)
    
    def __init__(self, world: World, map_canvas: tk.Canvas):
        self.world = world
        self.map_canvas = map_canvas
        
        # Occupancy Grid Set: Stores (row_i, col_j) of occupied cells.
        self.occupied_cells: set[Tuple[int, int]] = set() 
        
        # NN DATA STORAGE: Stores (State, Action, Reward) tuples.
        self.training_data: List[Tuple] = [] 
        
        # Movement State (defined in Pixels per Second for stability)
        self.move_speed_px = 180.0  # ~3.6 ft/s
        self.wall_follow_distance_mm = 600.0  # Target distance
        self.follow_dist_px = self.wall_follow_distance_mm / FT_TO_MM * FEET_TO_PX
        
        # Ground Truth Map (for scoring) - calculated once at initialization
        self.ground_truth_cells: set[Tuple[int, int]] = self._generate_ground_truth_map()
        
    def _px_to_grid(self, x_px: float, y_px: float) -> Optional[Tuple[int, int]]:
        """Converts pixel coordinates to a grid (row, col) index."""
        i = int(y_px // self.GRID_CELL_SIZE_PX) # Row (Y)
        j = int(x_px // self.GRID_CELL_SIZE_PX) # Col (X)
        if 0 <= i < self.WORLD_GRID_H and 0 <= j < self.WORLD_GRID_W:
            return i, j
        return None
    
    def _generate_ground_truth_map(self) -> set[Tuple[int, int]]:
        """
        Creates a high-resolution map of all actual shape/wall boundaries 
        to use as a comparison for scoring. (Simulation Ground Truth)
        """
        ground_truth: set[Tuple[int, int]] = set()
        
        for seg in self.world.all_segments():
            p0, p1 = seg
            x0, y0 = p0
            x1, y1 = p1
            
            # Sample along the segment
            length_px = math.hypot(x1 - x0, y1 - y0)
            # Sample every half cell size
            num_steps = max(2, int(length_px / (self.GRID_CELL_SIZE_PX * 0.5))) 
            
            for step in range(num_steps + 1):
                t = step / num_steps
                px = x0 + t * (x1 - x0)
                py = y0 + t * (y1 - y0)
                coords = self._px_to_grid(px, py)
                if coords:
                    # Add point and its immediate neighbors for thickness/rounding
                    r, c = coords
                    for dr in [-1, 0, 1]:
                         for dc in [-1, 0, 1]:
                            if 0 <= r+dr < self.WORLD_GRID_H and 0 <= c+dc < self.WORLD_GRID_W:
                                ground_truth.add((r+dr, c+dc))

        return ground_truth

    def calculate_map_score(self) -> Tuple[float, float, float]:
        """
        Compares the generated map to the ground truth map and returns scores.
        Returns: (Recall (Coverage), Precision, F1 Score (Accuracy))
        """
        GT = self.ground_truth_cells
        OCC = self.occupied_cells

        TP = len(OCC.intersection(GT)) # True Positives: Correctly mapped
        FP = len(OCC.difference(GT))  # False Positives: Mapped where empty (noise)
        FN = len(GT.difference(OCC))  # False Negatives: True obstacle missed (low coverage)
        
        # Recall (Coverage): How much of the truth did we find?
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # Precision: Of what we found, how much was correct?
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        # F1 Score (Accuracy): Harmonic mean of Precision and Recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return recall, precision, f1_score

    def record_training_step(self, distances_mm: List[int], turn_rate: float, forward_speed_px: float):
        """
        Records the current state, action, and calculated reward for NN training.
        """
        # 1. STATE (Input to the NN): Raw LiDAR distances
        current_state = distances_mm 
        
        # 2. ACTION (Output of the NN): The movement decision
        action_taken = (turn_rate, forward_speed_px)
        
        # 3. REWARD (Feedback for the NN)
        FRONT_INDEX = 0
        FRONT_SAFETY_MM = 500.0
        WALL_FOLLOW_INDEX = int(90.0 / DEG_PER_SAMPLE) % NUM_SAMPLES 
        target_dist_mm = self.wall_follow_distance_mm
        
        reward = 0.0
        
        # a) Collision Penalty/Movement Reward
        if distances_mm[FRONT_INDEX] < FRONT_SAFETY_MM:
            reward -= 5.0 # High penalty for being too close
        else:
            reward += 0.1 # Small reward for moving freely
            
        # b) Wall-Following Goal Reward
        side_dist_mm = distances_mm[WALL_FOLLOW_INDEX]
        error = abs(side_dist_mm - target_dist_mm) / target_dist_mm
        
        reward += max(0.0, 1.0 - error) * 0.5 
        
        # Store the training sample
        self.training_data.append((current_state, action_taken, reward))

    def process_data_string(self, framed_csv: str) -> List[int]:
        """Parses data and updates the Occupancy Grid Set."""
        
        if not (framed_csv.startswith(START_TOKEN) and framed_csv.endswith(END_TOKEN + "\n")):
             return []
             
        csv_data = framed_csv.strip(START_TOKEN + END_TOKEN + "\n")
        
        try:
            distances_mm = [int(x) for x in csv_data.split(",")]
        except ValueError:
            return []

        lx, ly = self.world.lidar_pos
        heading = self.world.lidar_heading_deg
        
        # Update Map Points (Convert to Grid Cell Coordinates)
        for i, mm in enumerate(distances_mm):
            dist_px = mm / FT_TO_MM * FEET_TO_PX
            
            angle_deg = heading - i * DEG_PER_SAMPLE
            th = math.radians(angle_deg)
            dx, dy = math.cos(th), math.sin(th)
            
            hit_x = lx + dx * dist_px
            hit_y = ly + dy * dist_px
            
            if 0 < hit_x < WORLD_PX and 0 < hit_y < WORLD_PX:
                coords = self._px_to_grid(hit_x, hit_y)
                if coords:
                    self.occupied_cells.add(coords) 
        
        return distances_mm

    def apply_movement_logic(self, distances_mm: List[int], dt: float):
        """
        Calculates the Action (turn_rate, forward_speed), records the step, and executes movement.
        Includes a check for colliding with the *mapped* occupied cells (new requirement).
        """
        if not distances_mm: return

        x, y = self.world.lidar_pos
        heading = self.world.lidar_heading_deg
        
        # --- 1. ACTION CALCULATION (Wall Follow Policy) ---
        forward_speed_px = self.move_speed_px
        turn_rate = 0.0
        
        WALL_FOLLOW_INDEX = int(90.0 / DEG_PER_SAMPLE) % NUM_SAMPLES 
        FRONT_INDEX = 0
        FRONT_SAFETY_MM = 500.0
        SAFETY_DIST_PX = FRONT_SAFETY_MM / FT_TO_MM * FEET_TO_PX
        
        # a) Collision Avoidance (based on raw sensor data)
        front_dist_mm = distances_mm[FRONT_INDEX]
        front_dist_px = front_dist_mm / FT_TO_MM * FEET_TO_PX
        
        if front_dist_px < SAFETY_DIST_PX:
            forward_speed_px = 0.0 # Stop
            turn_rate = 90.0 # Emergency Turn 
        
        # b) Wall Following Logic
        else:
            try:
                wall_dist_mm = distances_mm[WALL_FOLLOW_INDEX]
                wall_dist_px = wall_dist_mm / FT_TO_MM * FEET_TO_PX
                error = wall_dist_px - self.follow_dist_px
                
                K_p = 0.5 
                correction = -error * K_p * 30.0 
                max_correction = 5.0 
                
                turn_rate = max(min(correction, max_correction), -max_correction)
                
            except IndexError:
                pass 
        
        # --- 2. MAPPED WALL COLLISION CHECK (New Implementation) ---
        
        # Predict the next position based on the current calculated movement
        predicted_heading = heading + turn_rate * dt
        th_pred = math.radians(predicted_heading)
        dx, dy = math.cos(th_pred), math.sin(th_pred)
        predicted_x = x + dx * forward_speed_px * dt 
        predicted_y = y + dy * forward_speed_px * dt

        # Check if the predicted position overlaps with an already mapped occupied cell
        predicted_coords = self._px_to_grid(predicted_x, predicted_y)
        if predicted_coords and predicted_coords in self.occupied_cells:
            # Collision with a mapped object detected! Override movement.
            forward_speed_px = 0.0 
            turn_rate = 90.0 # Turn away from the mapped obstacle

        # --- 3. RECORD TRAINING STEP ---
        self.record_training_step(distances_mm, turn_rate, forward_speed_px)
        
        # --- 4. EXECUTE MOVEMENT ---
        heading += turn_rate * dt
        
        th = math.radians(heading)
        dx, dy = math.cos(th), math.sin(th)
        
        x += dx * forward_speed_px * dt 
        y += dy * forward_speed_px * dt

        # Boundary check 
        margin = LIDAR_RADIUS + WALL_THICKNESS/2 + 2
        x = max(margin, min(WORLD_PX - margin, x))
        y = max(margin, min(WORLD_PX - margin, y))

        self.world.lidar_pos = (x, y)
        self.world.lidar_heading_deg = heading % 360.0
        
    def draw_map(self):
        """Draws map by iterating over the small set of occupied cells."""
        self.map_canvas.delete("all")
        
        CELL_SIZE = self.GRID_CELL_SIZE_PX
        
        # Draw the mapped points (Occupied cells)
        for i, j in self.occupied_cells:
            # Calculate pixel corners for the cell
            x0, y0 = j * CELL_SIZE, i * CELL_SIZE
            x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE
            
            # Draw a filled rectangle for the cell 
            self.map_canvas.create_rectangle(x0, y0, x1, y1, 
                                             fill="#33ff33", outline="") 
        
        # Draw the current LiDAR position
        lx, ly = self.world.lidar_pos
        self.map_canvas.create_oval(lx - LIDAR_RADIUS, ly - LIDAR_RADIUS, 
                                    lx + LIDAR_RADIUS, ly + LIDAR_RADIUS,
                                    outline="#ff00ff", width=2) 

    def step(self, framed_csv: str, dt: float):
        distances_mm = self.process_data_string(framed_csv)
        self.apply_movement_logic(distances_mm, dt)
        self.draw_map()


# 6. UI (Tkinter) & Core Loop


WALL_THICKNESS = 6          # bold boundary
LIDAR_RADIUS = 10           # visual radius
RAY_HIGHLIGHT_WIDTH = 2.5   # index-0 ray thickness

INSTRUCTIONS = [
    "Controls:",
    "   Arrow keys = move (screen space)",
    "   A/D = rotate left/right   Q/E = fine rotate",
    "   Enter = print framed CSV (^ ... $) to console",
    " ",
    "Mapper Active (20s Run):",
    "   Movement is autonomous (wall follow).",
    "   Green squares show the accumulated Occupancy Grid map (high resolution).",
    "   NN Training Data is being collected in self.training_data list."
]

def load_shapes_from_config(world: World):
    world.rectangles.clear(); world.circles.clear()
    for r in SHAPES_CONFIG.get("rectangles", []):
        cx = r["cx_ft"] * FEET_TO_PX; cy = r["cy_ft"] * FEET_TO_PX
        w  = r["w_ft"]  * FEET_TO_PX; h  = r["h_ft"]  * FEET_TO_PX
        world.rectangles.append(Rectangle(cx, cy, w, h))
    for c in SHAPES_CONFIG.get("circles", []):
        cx = c["cx_ft"] * FEET_TO_PX; cy = c["cy_ft"] * FEET_TO_PX
        rr = c["r_ft"]  * FEET_TO_PX
        world.circles.append(Circle(cx, cy, rr))

class App:
    def __init__(self, use_mapper_movement=True) -> None: 
        self.root = tk.Tk()
        self.root.title("LiDAR Sandbox with Autonomous Mapping (NN Prep)")
        self.root.configure(bg="#0a0f1c")

        canvas_container = tk.Frame(self.root, width=WORLD_PX, height=WORLD_PX, bg="#0a0f1c")
        canvas_container.grid(row=0, column=0, padx=(12,6), pady=12)
        canvas_container.grid_propagate(False)

        self.canvas = tk.Canvas(canvas_container, width=WORLD_PX, height=WORLD_PX,
                                 bg="#0a0f1c", highlightthickness=0)
        self.canvas.place(x=0, y=0, anchor="nw") 

        self.map_overlay = tk.Canvas(canvas_container, width=WORLD_PX, height=WORLD_PX,
                                      bg='#0a0f1c', highlightthickness=0)
        self.map_overlay.place(x=0, y=0, anchor="nw")
        
        side = tk.Frame(self.root, bg="#101729")
        side.grid(row=0, column=1, sticky="ns", padx=(6,12), pady=12)

        tk.Label(side, text="LiDAR Sandbox", fg="white", bg="#101729",
                      font=("Segoe UI", 14, "bold")).pack(padx=12, pady=(12,4), anchor="w")
        
        # New: Timer Label
        self.TIMER_LIMIT_S = 20.0
        self.timer_label = tk.Label(side, text=f"Time Left: {self.TIMER_LIMIT_S:.2f}s", fg="#e6ff6b", bg="#101729",
                      font=("Segoe UI", 12, "bold"))
        self.timer_label.pack(padx=12, pady=(4,8), anchor="w")

        # New: Result Label
        self.result_label = tk.Label(side, text="", fg="#50ff50", bg="#101729",
                      font=("Segoe UI", 10, "bold"), justify=tk.LEFT)
        self.result_label.pack(padx=12, anchor="w")
        
        for line in INSTRUCTIONS:
            tk.Label(side, text=line, fg="#cdd7e7", bg="#101729",
                      font=("Consolas", 10)).pack(padx=12, anchor="w")

        tk.Label(side, text="Current frame (framed CSV):", fg="#cdd7e7", bg="#101729",
                      font=("Segoe UI", 10, "bold")).pack(padx=12, pady=(10,2), anchor="w")

        csv_frame = tk.Frame(side, bg="#101729")
        csv_frame.pack(padx=12, pady=(0,8), fill="x")

        self.csv_box = tk.Text(csv_frame, width=52, height=14, wrap="word",
                               bg="#0f1730", fg="#dfe7ff", insertbackground="#dfe7ff",
                               font=("Consolas", 9))
        self.csv_box.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(csv_frame, command=self.csv_box.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.csv_box.configure(yscrollcommand=scroll.set)
        csv_frame.grid_columnconfigure(0, weight=1)

        btns = tk.Frame(side, bg="#101729")
        btns.pack(padx=12, pady=(0,12), fill="x")
        ttk.Button(btns, text="Copy CSV to clipboard", command=self.copy_csv).pack(side="left")

        self.world = World()
        load_shapes_from_config(self.world)
        self.use_mapper_movement = use_mapper_movement
        self.mapper = Mapper(self.world, self.map_overlay) # Mapper is now defined

        self.speed_px = 5.0
        self.rotate_deg = 3.0
        self.keys_down = set()
        
        self.start_time = time.time()
        self.is_running = use_mapper_movement
        self.final_forward_speed_px = 0.0

        self.root.bind("<KeyPress>", self.on_key_down)
        self.root.bind("<KeyRelease>", self.on_key_up)

        self.last_time = time.time()
        self.last_csv_update = 0.0
        self.current_csv = ""
        self.tick()

        self.root.mainloop()

    # -------------- Input/IO --------------
    def on_key_down(self, e):
        name = e.keysym
        self.keys_down.add(name)
        if name == "Return":
            frame = simulate_frame(self.world)
            csv = ",".join(str(x) for x in frame)
            framed = f"{START_TOKEN}{csv}{END_TOKEN}\n"
            print(framed, end="")
            print("# distances are to the first hit (walls/shapes) at each ray angle")
            sys.stdout.flush()

    def on_key_up(self, e):
        name = e.keysym
        if name in self.keys_down:
            self.keys_down.remove(name)

    def copy_csv(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.current_csv)
        self.root.update()

    # -------------- Simulation (User-controlled movement - unchanged) --------------
    def apply_controls(self, dt):
        if self.use_mapper_movement: return

        x, y = self.world.lidar_pos; heading = self.world.lidar_heading_deg
        kl = set(k.lower() for k in self.keys_down)
        if "a" in kl: heading += self.rotate_deg
        if "d" in kl: heading -= self.rotate_deg
        if "q" in kl: heading += self.rotate_deg * 0.25
        if "e" in kl: heading -= self.rotate_deg * 0.25

        speed = self.speed_px * (2.0 if ("Shift_L" in self.keys_down or "Shift_R" in self.keys_down) else 1.0)
        if "Up" in self.keys_down: y -= speed
        if "Down" in self.keys_down: y += speed
        if "Left" in self.keys_down: x -= speed
        if "Right" in self.keys_down: x += speed

        margin = LIDAR_RADIUS + WALL_THICKNESS/2 + 2
        x = max(margin, min(WORLD_PX - margin, x))
        y = max(margin, min(WORLD_PX - margin, y))

        self.world.lidar_pos = (x, y)
        self.world.lidar_heading_deg = heading % 360.0

    # -------------- Rendering (unchanged) --------------
    def draw(self, frame_mm: List[int]):
        c = self.canvas
        c.delete("all")

        # Draw Grid
        for ft in range(1, int(WORLD_FT)):
            p = ft * FEET_TO_PX
            c.create_line(p, 0, p, WORLD_PX, fill="#25314f"); c.create_line(0, p, WORLD_PX, p, fill="#25314f")
        
        # Draw Boundary Wall
        c.create_rectangle(2, 2, WORLD_PX-2, WORLD_PX-2, outline="#59a5d8", width=WALL_THICKNESS)
        
        # Draw Rectangles
        for r in self.world.rectangles:
            x0 = r.cx - r.w/2.0; y0 = r.cy - r.h/2.0; x1 = r.cx + r.w/2.0; y1 = r.cy + r.h/2.0
            c.create_rectangle(x0, y0, x1, y1, outline="#b7e0ff")
            
        # Draw Circles
        for circ in self.world.circles:
            c.create_oval(circ.cx - circ.r, circ.cy - circ.r, circ.cx + circ.r, circ.cy + circ.r, outline="#ffd27f")

        lx, ly = self.world.lidar_pos; heading = self.world.lidar_heading_deg
        
        # Draw LiDAR Body
        c.create_oval(lx - LIDAR_RADIUS, ly - LIDAR_RADIUS, lx + LIDAR_RADIUS, ly + LIDAR_RADIUS,
                      fill="#e6ff6b", outline="#364a1a", width=2)
        # Draw Heading Indicator
        th0 = math.radians(heading); hx = lx + math.cos(th0) * 28; hy = ly + math.sin(th0) * 28
        c.create_line(lx, ly, hx, hy, fill="#364a1a", width=2)

        # Draw Rays
        for i in range(NUM_SAMPLES):
            angle_deg = heading - i*DEG_PER_SAMPLE
            th = math.radians(angle_deg)
            mm = frame_mm[i]; px = mm / FT_TO_MM * FEET_TO_PX
            ex = lx + math.cos(th) * px; ey = ly + math.sin(th) * px

            if i == 0:
                c.create_line(lx, ly, ex, ey, fill="#9ec3ff", width=RAY_HIGHLIGHT_WIDTH)
                c.create_text(ex, ey - 10, fill="#cdd7e7", text=f"{mm} mm", font=("Consolas", 10))
            else:
                c.create_line(lx, ly, ex, ey, fill="#6aa0ff", width=1)

        px_to_ft = 1.0 / FEET_TO_PX
        c.create_text(8, WORLD_PX - 16, anchor="w", fill="#cdd7e7",
                      text=f"Pos: ({lx*px_to_ft:.2f} ft, {ly*px_to_ft:.2f} ft)  Heading: {heading:05.1f}°  [Index 0 = front]")

    def update_csv_sidepanel(self, frame_mm: List[int]):
        now = time.time()
        if now - self.last_csv_update < 0.1: return
        self.last_csv_update = now

        csv_line = ",".join(str(x) for x in frame_mm)
        framed = f"{START_TOKEN}{csv_line}{END_TOKEN}\n"
        self.current_csv = framed
        self.csv_box.configure(state="normal")
        self.csv_box.delete("1.0", "end")
        self.csv_box.insert("1.0", framed)
        self.csv_box.configure(state="disabled")

    # -------------- Main loop --------------
    def tick(self):
        now = time.time()
        dt = min(0.05, now - self.last_time) if hasattr(self, "last_time") else 0.016
        self.last_time = now
        
        # --- Timer Management and Scoring ---
        if self.use_mapper_movement and self.is_running:
            elapsed = now - self.start_time
            time_left = self.TIMER_LIMIT_S - elapsed
            
            if time_left <= 0:
                self.is_running = False
                time_left = 0.0
                
                # Calculate and display score/speed upon stopping
                recall, precision, f1_score = self.mapper.calculate_map_score()
                
                # Convert final speed from px/s to ft/s
                final_speed_ft_s = (self.final_forward_speed_px / FEET_TO_PX)
                
                self.result_label.configure(
                    text=f"Time Up! Simulation Ended.\n"
                         f"  Map Coverage (Recall): {recall*100:.1f}%\n"
                         f"  Map Accuracy (F1 Score): {f1_score*100:.1f}%\n"
                         f"  Final Speed: {final_speed_ft_s:.2f} ft/s",
                    fg="#ffdd55"
                )

                self.timer_label.configure(text="Finished.")
            else:
                self.timer_label.configure(text=f"Time Left: {time_left:.2f}s")
        # --- End Timer Management ---


        frame_mm = simulate_frame(self.world)
        csv_line = ",".join(str(x) for x in frame_mm)
        framed_csv = f"{START_TOKEN}{csv_line}{END_TOKEN}\n"

        if self.use_mapper_movement and self.is_running:
            # Capture position before and after step to calculate speed
            old_pos = self.world.lidar_pos
            self.mapper.step(framed_csv, dt)
            new_pos = self.world.lidar_pos
            
            # Calculate actual distance moved and record speed
            distance_moved_px = math.hypot(new_pos[0]-old_pos[0], new_pos[1]-old_pos[1])
            self.final_forward_speed_px = distance_moved_px / dt if dt > 0 else 0.0
        else:
            self.apply_controls(dt)

        self.draw(frame_mm)
        self.current_csv = framed_csv
        self.update_csv_sidepanel(frame_mm)
        
        self.root.after(16, self.tick)

if __name__ == "__main__":
    # Runs the NN-prep autonomous mapper by default with the timer:
    App(use_mapper_movement=True)