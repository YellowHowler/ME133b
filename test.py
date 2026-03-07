import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt, atan2, cos, sin
from shapely.geometry import Point, LineString, Polygon

# Parameters
DSTEP = 0.5
DT = 0.4
SMAX = 40000
NMAX = 4000

GOAL_BIAS = 0.06
WAIT_PROB = 0.22  # waiting helps pass through a gap that opens/closes

xmin, xmax = 0, 10
ymin, ymax = 0, 12


# Split/Together Wall (one wall that becomes two segments)
class SplitWall:
    def __init__(self, y0=6.0, xmid=5.0, thickness=0.7, margin=0.6,
                 gmax=1, omega=0.28):
        self.y0 = y0
        self.xmid = xmid
        self.thickness = thickness
        self.margin = margin
        self.gmax = gmax
        self.omega = omega  # smaller => slower

    def gap(self, t):
        # smooth: 0 -> gmax -> 0
        return self.gmax * (np.sin(self.omega * t) ** 2)

    def polygons_at(self, t):
        g = self.gap(t)
        xL0 = xmin + self.margin
        xR1 = xmax - self.margin

        left_end  = self.xmid - g / 2.0
        right_beg = self.xmid + g / 2.0

        y0 = self.y0
        h = self.thickness
        y_bot = y0 - h/2
        y_top = y0 + h/2

        polys = []

        if left_end > xL0:
            polys.append(Polygon([
                (xL0,      y_bot),
                (left_end, y_bot),
                (left_end, y_top),
                (xL0,      y_top)
            ]))

        if xR1 > right_beg:
            polys.append(Polygon([
                (right_beg, y_bot),
                (xR1,       y_bot),
                (xR1,       y_top),
                (right_beg, y_top)
            ]))

        return polys


wall = SplitWall()


# Node
class Node:
    def __init__(self, x, y, t):
        self.x = float(x)
        self.y = float(y)
        self.t = float(t)
        self.parent = None

    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def inFreespace(self):
        if self.x <= xmin or self.x >= xmax or self.y <= ymin or self.y >= ymax:
            return False
        p = Point(self.x, self.y)
        for poly in wall.polygons_at(self.t):
            if not poly.disjoint(p):
                return False
        return True

    def connectsTo(self, other):
        #check wall at time of the new node
        line = LineString([(self.x, self.y), (other.x, other.y)])
        for poly in wall.polygons_at(self.t):
            if not poly.disjoint(line):
                return False
        return True


# Visualization
class Visualization:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.setup()

    def setup(self):
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

    def clear(self):
        self.ax.cla()
        self.setup()

    def draw_wall(self, t):
        for poly in wall.polygons_at(t):
            x, y = poly.exterior.xy
            self.ax.plot(x, y, 'k', linewidth=2)

        self.ax.plot([wall.xmid, wall.xmid], [ymin, ymax], linestyle='--', linewidth=1)
        self.ax.text(0.2, ymax - 0.6, f"t={t:.2f}, gap={wall.gap(t):.2f}", fontsize=10)

    def draw_node(self, node, **kwargs):
        self.ax.plot(node.x, node.y, **kwargs)

    def draw_edge(self, a, b, **kwargs):
        self.ax.plot([a.x, b.x], [a.y, b.y], **kwargs)

    def pause(self):
        plt.pause(0.001)


# Helpers
def build_path(goal):
    path = [goal]
    while path[0].parent:
        path.insert(0, path[0].parent)
    return path

def draw_final_snapshot(path, tree, visual, start, goal):
    tf = path[-1].t
    visual.clear()
    visual.draw_wall(tf)

    # tree
    for n in tree:
        if n.parent:
            visual.draw_edge(n.parent, n, color='lightgreen', linewidth=0.5)

    # final path
    for i in range(len(path)-1):
        visual.draw_edge(path[i], path[i+1], color='red', linewidth=3)

    visual.draw_node(start, color='orange', marker='o')
    visual.draw_node(goal,  color='purple', marker='o')
    visual.pause()


# Temporal RRT
def rrt(start, goal, visual):
    tree = [start]
    steps = 0

    while steps < SMAX and len(tree) < NMAX:
        steps += 1

        if random.random() < GOAL_BIAS:
            target = goal
        else:
            target = Node(random.uniform(xmin, xmax),
                          random.uniform(ymin, ymax),
                          0.0)

        nearest = min(tree, key=lambda n: n.distance(target))

        if random.random() < WAIT_PROB:
            newn = Node(nearest.x, nearest.y, nearest.t + DT)  # wait
        else:
            ang = atan2(target.y - nearest.y, target.x - nearest.x)
            newn = Node(nearest.x + DSTEP*cos(ang),
                        nearest.y + DSTEP*sin(ang),
                        nearest.t + DT)

        if newn.inFreespace() and newn.connectsTo(nearest):
            newn.parent = nearest
            tree.append(newn)

            visual.clear()
            visual.draw_wall(newn.t)
            for n in tree:
                if n.parent:
                    visual.draw_edge(n.parent, n, color='green', linewidth=0.5)
            visual.draw_node(start, color='orange', marker='o')
            visual.draw_node(goal,  color='purple', marker='o')
            visual.pause()

            if newn.distance(goal) < DSTEP:
                goal_reached = Node(goal.x, goal.y, newn.t + DT)
                if goal_reached.inFreespace() and goal_reached.connectsTo(newn):
                    goal_reached.parent = newn
                    return build_path(goal_reached), tree

    return None, tree


# Animation
def animate(path, visual, start, goal):
    tf = path[-1].t
    t = 0.0
    t_step = 0.1      
    frame_delay = 0.01 

    print(f"Animating... simulated tf={tf:.2f}s, frames ~ {int(tf/t_step)+1}")

    trail_x, trail_y = [], []

    while t <= tf:
        visual.clear()
        visual.draw_wall(t)

        # draw planned path
        for i in range(len(path)-1):
            visual.draw_edge(path[i], path[i+1], color='red', linewidth=2)

        # interpolate
        x, y = path[0].x, path[0].y
        for i in range(len(path)-1):
            if path[i].t <= t <= path[i+1].t:
                denom = path[i+1].t - path[i].t
                a = 0.0 if denom <= 1e-9 else (t - path[i].t) / denom
                x = path[i].x + a * (path[i+1].x - path[i].x)
                y = path[i].y + a * (path[i+1].y - path[i].y)
                break

        # draw trajectory
        trail_x.append(x)
        trail_y.append(y)
        visual.ax.plot(trail_x, trail_y, linewidth=2)
        visual.ax.plot([x], [y], 'bo', markersize=6)
        visual.draw_node(start, color='orange', marker='o') # mark start
        visual.draw_node(goal,  color='purple', marker='o') # mark goal

        plt.pause(frame_delay)
        t += t_step


# Main
def main():
    visual = Visualization()

    start = Node(wall.xmid, 10.5, 0.0)
    goal  = Node(wall.xmid,  1.5, 0.0)

    if not start.inFreespace():
        print("Start is in collision at t=0. Adjust start.")
        return
    if not goal.inFreespace():
        print("Goal is in collision at t=0. Adjust goal.")
        return

    print("Planning...")
    path, tree = rrt(start, goal, visual)
    if path is None:
        print("Failed. Try increasing SMAX/NMAX or WAIT_PROB or gmax.")
        return

    print(f"Path found. tf={path[-1].t:.2f}, nodes={len(tree)}")

    # show final planning map
    draw_final_snapshot(path, tree, visual, start, goal)
    input("Final path shown (with wall at final time). Press Enter to animate...")

    # animation
    animate(path, visual, start, goal)
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()