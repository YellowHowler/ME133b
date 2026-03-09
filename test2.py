import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon
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

XMIN, XMAX = 0, 6
YMIN, YMAX = 0, 7

WALL_THICKNESS = 0.1
WALL_SPLIT_OMEGA = 0.28

######################################################################
#
# Wall: Default non-moving wall
class Wall:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    
    def polygonsAt(self, t):
        return [Polygon([
            (self.xmin, self.ymin),
            (self.xmax, self.ymin),
            (self.xmax, self.ymax),
            (self.xmin, self.ymax)
        ])]

######################################################################
#
# Splitting Wall (inherits Wall): One wall that becomes two segments in the x direction
class SplitWall(Wall):
    def __init__(self, xmin, xmax, ymin, ymax, gappos, gapmax, omega, splitdir=0):
        super().__init__(xmin, xmax, ymin, ymax)
        self.gappos = gappos # x coordinate of the center of the gap
        self.splitdir = splitdir # 0: x, 1: y
        self.gapmax = gapmax
        self.omega = omega # smaller => slower

    def gap(self, t):
        # smooth: 0 -> gapmax -> 0
        return self.gapmax * (np.sin(self.omega * t) ** 2)

    def polygonsAt(self, t):
        g = self.gap(t)
        polys = []

        if self.splitdir == 0: # split in x direction
            xL0 = self.xmin
            xR1 = self.xmax

            xR0  = self.gappos - g / 2.0
            xL1 = self.gappos + g / 2.0

            yB = self.ymin
            yT = self.ymax

            if xR0 > xL0:
                polys.append(Polygon([
                    (xL0, yB),
                    (xR0, yB),
                    (xR0, yT),
                    (xL0, yT)
                ]))

            if xR1 > xL1:
                polys.append(Polygon([
                    (xL1, yB),
                    (xR1, yB),
                    (xR1, yT),
                    (xL1, yT)
                ]))
        else: # split in y direction
            yB0 = self.ymin
            yT1 = self.ymax

            yT0  = self.gappos - g / 2.0
            yB1 = self.gappos + g / 2.0

            xL = self.xmin
            xR = self.xmax

            if yT0 > yB0:
                polys.append(Polygon([
                    (xL, yB0),
                    (xR, yB0),
                    (xR, yT0),
                    (xL, yT0)
                ]))

            if yT1 > yB1:
                polys.append(Polygon([
                    (xL, yB1),
                    (xR, yB1),
                    (xR, yT1),
                    (xL, yT1)
                ]))

        return polys

class Map():
    def __init__(self, walls, xmin, xmax, ymin, ymax):
        self.walls = walls
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def polygonsAt(self, t):
        polys = []
        for wall in self.walls:
            polys.extend(wall.polygonsAt(t)) 
        return polys

# Node
class Node:
    def __init__(self, x, y, t, map):
        self.x = float(x)
        self.y = float(y)
        self.t = float(t)
        self.parent = None
        self.map = map

    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def inFreespace(self):
        if self.x <= XMIN or self.x >= XMAX or self.y <= YMIN or self.y >= YMAX:
            return False
        p = Point(self.x, self.y)
        for poly in self.map.polygonsAt(self.t):
            if not poly.disjoint(p):
                return False
        return True

    def connectsTo(self, other):
        #check wall at time of the new node
        line = LineString([(self.x, self.y), (other.x, other.y)])
        for poly in self.map.polygonsAt(other.t):
            if not poly.disjoint(line):
                return False
        return True


######################################################################
#
#   Visualization Class
#
#   This renders the world.  In particular it provides the methods:
#     show(text = '')                   Show the current figure
#     drawNode(node,         **kwargs)  Draw a single node
#     drawEdge(node1, node2, **kwargs)  Draw an edge between nodes
#     drawPath(path,         **kwargs)  Draw a path (list of nodes)
#
class Visualization:
    def __init__(self, map):
        self.map = map
        self.fig, self.ax = plt.subplots()
        self.drawMap(0.0)

    def drawMap(self, t):
        self.clear()

        self.ax.grid(True)
        self.ax.axis('on')
        self.ax.set_xlim(self.map.xmin, self.map.xmax)
        self.ax.set_ylim(self.map.ymin, self.map.ymax)
        self.ax.set_aspect('equal')

        for poly in self.map.polygonsAt(t):
            x, y = poly.exterior.xy
            patch = MplPolygon(list(zip(x, y)),
                           closed=True,
                           facecolor='black',
                           edgecolor='black')
            self.ax.add_patch(patch)

    def show(self, text=''):
        plt.pause(0.001)
        if len(text) > 0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, **kwargs):
        self.ax.plot(node.x, node.y, **kwargs)

    def drawEdge(self, head, tail, **kwargs):
        self.ax.plot([head.x, tail.x], [head.y, tail.y], **kwargs)

    def drawPath(self, path, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], **kwargs)

    def drawStartGoal(self, start, goal):

        self.drawNode(start, color='orange', marker='o')
        self.drawNode(goal,  color='purple', marker='o')

    def drawRobot(self, x, y):
        self.ax.plot(x, y, color='red', marker='o')

    def interpolatePath(self, path, t):
        px = [node.x for node in path]
        py = [node.y for node in path]
        pt = [node.t for node in path]

        if t <= pt[0]:
            return px[0], py[0]
        if t >= pt[-1]:
            return px[-1], py[-1]

        for i in range(len(pt) - 1):
            if pt[i] <= t <= pt[i + 1]:
                alpha = (t - pt[i]) / (pt[i + 1] - pt[i])
                x = px[i] + alpha * (px[i + 1] - px[i])
                y = py[i] + alpha * (py[i + 1] - py[i])
                return x, y

        return px[-1], py[-1]
    
    def draw_final_snapshot(self, path, tree, start, goal):
        tf = path[-1].t
        self.drawMap(tf)

        # tree
        for n in tree:
            if n.parent:
                self.drawEdge(n.parent, n, color='lightgreen', linewidth=0.5)

        # final path
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], color='red', linewidth=3)

        self.drawStartGoal(start, goal)
        self.show('Final snapshot (hit return to continue)')
    
    def drawFrame(self, path, start, goal, t):
        self.drawMap(t)
        self.drawPath(path, color='red', linewidth=2)
        self.drawStartGoal(start, goal)
        x, y = self.interpolatePath(path, t)
        self.drawRobot(x, y)

    def clear(self):
        self.ax.clear()

# Helpers
def build_path(goal):
    path = [goal]
    while path[0].parent:
        path.insert(0, path[0].parent)
    return path


# Temporal RRT
def rrt(start, goal, visual):
    map = visual.map

    tree = [start]
    steps = 0

    def addToTree(oldn, newn):
        newn.parent = oldn
        tree.append(newn)

        visual.drawEdge(oldn, newn, color='g', linewidth=1)
        visual.show()

    while steps < SMAX and len(tree) < NMAX:
        steps += 1

        if random.random() < GOAL_BIAS:
            target = goal
        else:
            target = Node(random.uniform(map.xmin, map.xmax),
                          random.uniform(map.ymin, map.ymax),
                          0.0,
                          map)

        nearest = min(tree, key=lambda n: n.distance(target))

        if random.random() < WAIT_PROB:
            newn = Node(nearest.x, nearest.y, nearest.t + DT, map)  # wait
        else:
            ang = atan2(target.y - nearest.y, target.x - nearest.x)
            newn = Node(nearest.x + DSTEP*cos(ang),
                        nearest.y + DSTEP*sin(ang),
                        nearest.t + DT,
                        map)

        if newn.inFreespace() and newn.connectsTo(nearest):
            addToTree(nearest, newn)

            if newn.distance(goal) < DSTEP:
                goal_reached = Node(goal.x, goal.y, newn.t + DT, map)
                if goal_reached.inFreespace() and goal_reached.connectsTo(newn):
                    goal_reached.parent = newn
                    return build_path(goal_reached), tree

    return None, tree

def postProcess(path):
    shortpath = [path[0]]
    for i in range(2, len(path)):
        if not shortpath[-1].connectsTo(path[i]):
            shortpath.append(path[i-1])
    shortpath.append(path[-1])
    return shortpath

def animation(path, visual, start, goal):
    tf = path[-1].t
    fps = 30
    duration = int(tf * 0.4)
    
    def update(frame):
        t = tf * frame / (duration * fps - 1)
        visual.drawFrame(path, start, goal, t)
        return []

    ani = FuncAnimation(
        visual.fig,
        update,
        frames=duration * fps,
        interval=1000 / fps,
        blit=False,
        repeat=False 
    )

    return ani

# Main
def main():
    def H(x1, x2, y):
        return Wall(min(x1, x2), max(x1, x2), y - WALL_THICKNESS/2, y + WALL_THICKNESS/2)
    def V(x, y1, y2):
        return Wall(x - WALL_THICKNESS/2, x + WALL_THICKNESS/2, min(y1, y2), max(y1, y2))
    def HM(x1, x2, y):
        return SplitWall(min(x1, x2), max(x1, x2), y - WALL_THICKNESS/2, y + WALL_THICKNESS/2,
                         gappos=(x1+x2)/2, gapmax=1, omega=WALL_SPLIT_OMEGA)
    def VM(x, y1, y2):
        return SplitWall(x - WALL_THICKNESS/2, x + WALL_THICKNESS/2, min(y1, y2), max(y1, y2),
                         gappos=(y1+y2)/2, gapmax=1, omega=WALL_SPLIT_OMEGA, splitdir=1)

    walls = [
        # Outer border
        H(0, 6, 7),
        H(0, 6, 0),
        V(0, 0, 7),
        V(6, 0, 7),

        # Inner walls
        H(1, 5, 6),
        H(1, 2, 5),
        H(3, 5, 5),
        H(2, 3, 4),
        HM(4, 6, 4),
        H(0, 1, 3),
        H(2, 5, 3),
        H(0, 4, 2),
        HM(2, 3, 1),
        H(4, 5, 1,),
        V(1, 0, 1),
        V(1, 3, 5),
        VM(2, 4, 5),
        V(3, 1, 2),
        VM(4, 0, 1),
        V(4, 4, 5),
        VM(5, 1, 3),
        V(5, 5, 6)
    ]
    map = Map(walls, XMIN, XMAX, YMIN, YMAX)

    visual = Visualization(map)

    start = Node(0.5, 0.5, 0.0, map)
    goal  = Node(4.5, 4.5, 0.0, map)

    visual.drawStartGoal(start, goal)
    visual.show("Showing basic world")

    if not start.inFreespace():
        print("Start is in collision at t=0. Adjust start.")
        return
    if not goal.inFreespace():
        print("Goal is in collision at t=0. Adjust goal.")
        return

    print("Planning...")
    path, tree = rrt(start, goal, visual)
    path = postProcess(path)
    if path is None:
        print("Failed. Try increasing SMAX/NMAX or WAIT_PROB or gmax.")
        return

    print(f"Path found. tf={path[-1].t:.2f}, nodes={len(tree)}")

    # show final planning map
    visual.draw_final_snapshot(path, tree, start, goal)
    input("Final path shown (with wall at final time). Press Enter to animate...")
    
    # show final animation
    plt.close(visual.fig)
    visual_ani = Visualization(map)
    ani = animation(path, visual_ani, start, goal)
    plt.show()

if __name__ == "__main__":
    main()