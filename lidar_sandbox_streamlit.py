import math
import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# World / Units

FT_TO_MM = 304.8
FEET_TO_PX = 50  # 1 ft = 50 px
WORLD_FT = 15.0
WORLD_PX = int(WORLD_FT * FEET_TO_PX)  # 750 px

NUM_SAMPLES = 240
DEG_PER_SAMPLE = 360.0 / NUM_SAMPLES  # 1.5°

# Serial framing (for easy “beginning” / “end” detection)
START_TOKEN = "^"
END_TOKEN   = "$"

# Shapes config (EDIT THIS)
# Units here are in FEET for convenience

SHAPES_CONFIG = {
    # Rectangles defined by center (x_ft, y_ft) and size (w_ft, h_ft)
    "rectangles": [
        # Example shapes — uncomment/modify/add:
        {"cx_ft": 7.5, "cy_ft": 10.0, "w_ft": 3.0, "h_ft": 1.5},
        {"cx_ft": 5.0,  "cy_ft": 6.0,  "w_ft": 2.0, "h_ft": 2.0},
    ],
    # Circles defined by center (x_ft, y_ft) and radius (r_ft)
    "circles": [
        {"cx_ft": 10.0, "cy_ft": 5.0, "r_ft": 1.0},
    ],
}

# Geometry primitives

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

@dataclass
class Rectangle:
    cx: float
    cy: float
    w: float
    h: float
    def segments(self) -> List[Segment]:
        x0 = self.cx - self.w/2.0
        x1 = self.cx + self.w/2.0
        y0 = self.cy - self.h/2.0
        y1 = self.cy + self.h/2.0
        return [((x0,y0),(x1,y0)), ((x1,y0),(x1,y1)), ((x1,y1),(x0,y1)), ((x0,y1),(x0,y0))]

@dataclass
class Circle:
    cx: float
    cy: float
    r: float

@dataclass
class World:
    size_px: int = WORLD_PX
    lidar_pos: Point = field(default_factory=lambda: (WORLD_PX/2, WORLD_PX/2))
    # IMPORTANT: -90° makes index 0 point UP on screen
    lidar_heading_deg: float = -90.0
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


# Ray casting helpers

def ray_segment_intersection(ray_o: Point, ray_d: Point, p0: Point, p1: Point) -> Optional[float]:
    """Return distance t (in px) along ray O + t D, t>=0 to segment [p0,p1], or None."""
    ox, oy = ray_o
    dx, dy = ray_d
    x1, y1 = p0
    x2, y2 = p1
    vx, vy = x2 - x1, y2 - y1
    denom = dx * (-vy) + dy * (vx)
    if abs(denom) < 1e-9:
        return None
    rx, ry = x1 - ox, y1 - oy
    t = (rx * (-vy) + ry * (vx)) / denom
    if t < 0:
        return None
    ix = ox + t*dx
    iy = oy + t*dy
    u = ((ix - x1)*vx + (iy - y1)*vy) / (vx*vx + vy*vy)
    if u < 0 or u > 1:
        return None
    return t

def ray_circle_intersection(ray_o: Point, ray_d: Point, center: Point, r: float) -> Optional[float]:
    """Return nearest non-negative t (in px) where ray hits circle, or None."""
    ox, oy = ray_o
    dx, dy = ray_d
    cx, cy = center
    fx, fy = ox - cx, oy - cy
    a = dx*dx + dy*dy  # 1 if D is normalized
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    ts = [t for t in (t1, t2) if t >= 0]
    if not ts:
        return None
    return min(ts)

def simulate_frame(world: World) -> List[int]:
    """Return 240 distances in millimeters, first hit per ray (walls + shapes)."""
    lx, ly = world.lidar_pos
    distances_px: List[float] = [0.0]*NUM_SAMPLES
    segs = world.all_segments()

    for i in range(NUM_SAMPLES):
        # index 0 ray points along current heading; indices increase clockwise
        angle_deg = world.lidar_heading_deg - i*DEG_PER_SAMPLE
        th = math.radians(angle_deg)
        dx, dy = math.cos(th), math.sin(th)
        best_t: Optional[float] = None

        # Segments (walls + rectangles)
        for p0, p1 in segs:
            t = ray_segment_intersection((lx,ly), (dx,dy), p0, p1)
            if t is not None and (best_t is None or t < best_t):
                best_t = t

        # Circles
        for c in world.circles:
            t = ray_circle_intersection((lx,ly), (dx,dy), (c.cx,c.cy), c.r)
            if t is not None and (best_t is None or t < best_t):
                best_t = t

        if best_t is None:
            best_t = WORLD_PX * 2.0  # safety (should always hit a wall)

        distances_px[i] = best_t

    # Convert px -> mm
    px_to_mm = (1.0 / FEET_TO_PX) * FT_TO_MM
    distances_mm = [int(round(d * px_to_mm)) for d in distances_px]
    return distances_mm


# UI (Tkinter)

import tkinter as tk
from tkinter import ttk

WALL_THICKNESS = 6         # bold boundary
LIDAR_RADIUS = 10          # visual radius
RAY_HIGHLIGHT_WIDTH = 2.5  # index-0 ray thickness

INSTRUCTIONS = [
    "Controls:",
    "  Arrow keys = move (screen space)",
    "  A/D = rotate left/right   Q/E = fine rotate",
    "  Enter = print framed CSV (^ ... $) to console",
]

def load_shapes_from_config(world: World):
    """Populate world's rectangles/circles from SHAPES_CONFIG (feet -> pixels)."""
    world.rectangles.clear()
    world.circles.clear()
    for r in SHAPES_CONFIG.get("rectangles", []):
        cx = r["cx_ft"] * FEET_TO_PX
        cy = r["cy_ft"] * FEET_TO_PX
        w  = r["w_ft"]  * FEET_TO_PX
        h  = r["h_ft"]  * FEET_TO_PX
        world.rectangles.append(Rectangle(cx, cy, w, h))
    for c in SHAPES_CONFIG.get("circles", []):
        cx = c["cx_ft"] * FEET_TO_PX
        cy = c["cy_ft"] * FEET_TO_PX
        rr = c["r_ft"]  * FEET_TO_PX
        world.circles.append(Circle(cx, cy, rr))

class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("LiDAR Real-Time Sandbox (Full 2D world + shapes + framed CSV)")
        self.root.configure(bg="#0a0f1c")

        # Left: world canvas (static view of full world)
        self.canvas = tk.Canvas(self.root, width=WORLD_PX, height=WORLD_PX,
                                bg="#0a0f1c", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=(12,6), pady=12)

        # Right: side panel with instructions and live CSV
        side = tk.Frame(self.root, bg="#101729")
        side.grid(row=0, column=1, sticky="ns", padx=(6,12), pady=12)

        tk.Label(side, text="LiDAR Sandbox", fg="white", bg="#101729",
                 font=("Segoe UI", 14, "bold")).pack(padx=12, pady=(12,4), anchor="w")
        for line in INSTRUCTIONS:
            tk.Label(side, text=line, fg="#cdd7e7", bg="#101729",
                     font=("Consolas", 10)).pack(padx=12, anchor="w")

        # CSV box
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

        # World + shapes
        self.world = World()
        load_shapes_from_config(self.world)

        # Controls
        self.speed_px = 5.0
        self.rotate_deg = 3.0
        self.keys_down = set()

        # bindings
        self.root.bind("<KeyPress>", self.on_key_down)
        self.root.bind("<KeyRelease>", self.on_key_up)

        # draw loop
        self.last_time = time.time()
        self.last_csv_update = 0.0
        self.current_csv = ""
        self.tick()

        self.root.mainloop()

    # -------------- Input --------------
    def on_key_down(self, e):
        name = e.keysym
        self.keys_down.add(name)
        if name == "Return":
            frame = simulate_frame(self.world)
            # 1) framed CSV for serial
            csv = ",".join(str(x) for x in frame)
            framed = f"{START_TOKEN}{csv}{END_TOKEN}\n"
            print(framed, end="")
            # 2) (Optional) angle map comment for debugging
            annotated = "  ".join(
                f"[i={i:3d}, ang={self.world.lidar_heading_deg - i*DEG_PER_SAMPLE:7.2f}°]={d}mm"
                for i, d in enumerate(frame)
            )
            print("# distances are to the first hit (walls/shapes) at each ray angle")
            print("# " + annotated)
            sys.stdout.flush()

    def on_key_up(self, e):
        name = e.keysym
        if name in self.keys_down:
            self.keys_down.remove(name)

    def copy_csv(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.current_csv)
        self.root.update()

    # -------------- Simulation --------------
    def apply_controls(self, dt):
        x, y = self.world.lidar_pos
        heading = self.world.lidar_heading_deg

        # Rotation with A/D (coarse) and Q/E (fine)
        kl = set(k.lower() for k in self.keys_down)
        if "a" in kl:
            heading += self.rotate_deg
        if "d" in kl:
            heading -= self.rotate_deg
        if "q" in kl:
            heading += self.rotate_deg * 0.25
        if "e" in kl:
            heading -= self.rotate_deg * 0.25

        # Screen-space movement with arrow keys (↑ up)
        speed = self.speed_px * (2.0 if ("Shift_L" in self.keys_down or "Shift_R" in self.keys_down) else 1.0)
        if "Up" in self.keys_down:
            y -= speed
        if "Down" in self.keys_down:
            y += speed
        if "Left" in self.keys_down:
            x -= speed
        if "Right" in self.keys_down:
            x += speed

        # Keep LiDAR fully inside walls
        margin = LIDAR_RADIUS + WALL_THICKNESS/2 + 2
        x = max(margin, min(WORLD_PX - margin, x))
        y = max(margin, min(WORLD_PX - margin, y))

        self.world.lidar_pos = (x, y)
        self.world.lidar_heading_deg = heading % 360.0

    # Rendering
    def draw(self, frame_mm: List[int]):
        c = self.canvas
        c.delete("all")

        # Grid (1 ft spacing)
        c.configure(bg="#0a0f1c")
        for ft in range(1, int(WORLD_FT)):
            p = ft * FEET_TO_PX
            c.create_line(p, 0, p, WORLD_PX, fill="#25314f")
            c.create_line(0, p, WORLD_PX, p, fill="#25314f")

        # Walls/boundary (bold line)
        c.create_rectangle(2, 2, WORLD_PX-2, WORLD_PX-2, outline="#59a5d8", width=WALL_THICKNESS)

        # Draw configured rectangles
        for r in self.world.rectangles:
            x0 = r.cx - r.w/2.0
            y0 = r.cy - r.h/2.0
            x1 = r.cx + r.w/2.0
            y1 = r.cy + r.h/2.0
            c.create_rectangle(x0, y0, x1, y1, outline="#b7e0ff")

        # Draw configured circles
        for circ in self.world.circles:
            c.create_oval(circ.cx - circ.r, circ.cy - circ.r, circ.cx + circ.r, circ.cy + circ.r, outline="#ffd27f")

        # LiDAR + heading tick (index 0 points UP with heading=-90 by default)
        lx, ly = self.world.lidar_pos
        heading = self.world.lidar_heading_deg
        c.create_oval(lx - LIDAR_RADIUS, ly - LIDAR_RADIUS, lx + LIDAR_RADIUS, ly + LIDAR_RADIUS,
                      fill="#e6ff6b", outline="#364a1a", width=2)
        th0 = math.radians(heading)
        hx = lx + math.cos(th0) * 28
        hy = ly + math.sin(th0) * 28
        c.create_line(lx, ly, hx, hy, fill="#364a1a", width=2)

        # Rays (walls + shapes). Highlight index 0 and label its distance.
        for i in range(NUM_SAMPLES):
            angle_deg = heading - i*DEG_PER_SAMPLE
            th = math.radians(angle_deg)
            dx, dy = math.cos(th), math.sin(th)
            mm = frame_mm[i]
            px = mm / FT_TO_MM * FEET_TO_PX
            ex = lx + dx * px
            ey = ly + dy * px

            if i == 0:
                c.create_line(lx, ly, ex, ey, fill="#9ec3ff", width=RAY_HIGHLIGHT_WIDTH)
                c.create_text(ex, ey - 10, fill="#cdd7e7",
                              text=f"{mm} mm", font=("Consolas", 10))
            else:
                c.create_line(lx, ly, ex, ey, fill="#6aa0ff", width=1)

        # HUD
        px_to_ft = 1.0 / FEET_TO_PX
        c.create_text(8, WORLD_PX - 16, anchor="w", fill="#cdd7e7",
                      text=f"Pos: ({lx*px_to_ft:.2f} ft, {ly*px_to_ft:.2f} ft)  Heading: {heading:05.1f}°  [Index 0 = front]")

    def update_csv_sidepanel(self, frame_mm: List[int]):
        # Update at most ~10 Hz so the UI stays snappy
        now = time.time()
        if now - self.last_csv_update < 0.1:
            return
        self.last_csv_update = now

        csv_line = ",".join(str(x) for x in frame_mm)
        framed = f"{START_TOKEN}{csv_line}{END_TOKEN}\n"
        self.current_csv = framed
        self.csv_box.configure(state="normal")
        self.csv_box.delete("1.0", "end")
        self.csv_box.insert("1.0", framed)
        self.csv_box.configure(state="disabled")

    # Main loop 
    def tick(self):
        now = time.time()
        dt = min(0.05, now - self.last_time) if hasattr(self, "last_time") else 0.016
        self.last_time = now
        self.apply_controls(dt)
        frame_mm = simulate_frame(self.world)
        self.draw(frame_mm)
        self.update_csv_sidepanel(frame_mm)
        self.root.after(16, self.tick)

if __name__ == "__main__":
    App()
