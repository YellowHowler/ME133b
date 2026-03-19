import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Patch
import numpy as np
import random
from math import sqrt, atan2, cos, sin
from shapely.geometry import Point, LineString, Polygon

# Parameters
DSTEP = 0.6
DT = 0.4
SMAX = 40000
NMAX = 4000

CLEARANCE = 0.1

GOAL_BIAS = 0.1
WAIT_PROB = 0.22  # waiting helps pass through a gap that opens/closes

XMIN, XMAX = 0, 6
YMIN, YMAX = 0, 7

WALL_THICKNESS = 0.1
WALL_SPLIT_OMEGA = 0.28

D_ACC_MAX = 0.2
D_ANG_MAX = 0.5
ACC_MAX = 0.6
VEL_MAX = 1.5

######################################################################
#
# Wall: Default non-moving wall
class Wall:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    
    def polygonsAt(self, t, returnColor=False):
        poly = Polygon([
            (self.xmin, self.ymin),
            (self.xmax, self.ymin),
            (self.xmax, self.ymax),
            (self.xmin, self.ymax)
        ])

        if returnColor:
            return [(poly, 'black')]

        return [poly]

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

    def polygonsAt(self, t, returnColor=False):
        g = self.gap(t)
        polys = []

        def appendPoly(poly):
            if returnColor:
                polys.append((poly, 'orange'))
            else:
                polys.append(poly)

        if self.splitdir == 0: # split in x direction
            xL0 = self.xmin
            xR1 = self.xmax

            xR0  = self.gappos - g / 2.0
            xL1 = self.gappos + g / 2.0

            yB = self.ymin
            yT = self.ymax

            if xR0 > xL0:
                appendPoly(Polygon([
                    (xL0, yB),
                    (xR0, yB),
                    (xR0, yT),
                    (xL0, yT)
                ]))

            if xR1 > xL1:
                appendPoly(Polygon([
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
                appendPoly(Polygon([
                    (xL, yB0),
                    (xR, yB0),
                    (xR, yT0),
                    (xL, yT0)
                ]))

            if yT1 > yB1:
                appendPoly(Polygon([
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

    def polygonsAt(self, t, returnColor=False):
        polys = []
        for wall in self.walls:
            polys.extend(wall.polygonsAt(t, returnColor=returnColor))
        return polys

# Node
class Node:
    def __init__(self, x, y, t, map):
        self.x = float(x)
        self.y = float(y)
        self.t = float(t)
        self.parent = None
        self.child = None
        self.map = map
        self.cost = 0
        self.vel = 0
        self.acc = 0
        self.ang = 0

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

    def connectsTo(self, other, enforceDynamics=False):
        line = LineString([(self.x, self.y), (other.x, other.y)]) 
        for poly in self.map.polygonsAt(other.t): 
            if not poly.disjoint(line): 
                return False 
            if line.distance(poly) < CLEARANCE: 
                return False
        
        # check if max velocity / turn direction is violated
        def wrapTwoPi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        if enforceDynamics:
            dt = other.t - self.t
            if dt <= 0:
                return False

            if (self.distance(other) / dt) > VEL_MAX:
                return False

            ang_diff = wrapTwoPi(other.ang - self.ang)
            if abs(ang_diff) > D_ANG_MAX:
                return False
           
        return True
    
    def inOpenSpace(self, other, clearance=CLEARANCE):
        dt = other.t - self.t
        if dt <= 0:
            return False
        
        numCheck = 10
        for i in range(numCheck+1):
            alpha = i / numCheck
            x = self.x + alpha * (other.x - self.x)
            y = self.y + alpha * (other.y - self.y)
            t = self.t + alpha * dt

            p = Point(x, y)
            for poly in self.map.polygonsAt(t):
                if poly.distance(p) < clearance:
                    return False
                
        return True
    
    def numCloseWalls(self, radius):
        count = 0
        p = Point(self.x, self.y)

        for poly in self.map.polygonsAt(self.t):
            if poly.distance(p) < radius:
                count += 1

        return count

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

        for poly, color in self.map.polygonsAt(t, returnColor=True):
            x, y = poly.exterior.xy
            patch = MplPolygon(list(zip(x, y)),
                           closed=True,
                           facecolor=color,
                           edgecolor=color)
            self.ax.add_patch(patch)

        legend_elements = [
            Patch(facecolor='black', edgecolor='black', label='Still Wall'),
            Patch(facecolor='orange', edgecolor='orange', label='Splitting Wall')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

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
    
    def draw_final_snapshot(self, path, tree, start, goal, color='red'):
        tf = path[-1].t
        self.drawMap(tf)

        # tree
        for n in tree:
            if n.parent:
                self.drawEdge(n.parent, n, color='lightgreen', linewidth=0.5)

        # final path
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], color=color, linewidth=3)

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

# Temporal RRT*
def rrtstar(start, goal, visual):
    map = visual.map

    tree = [start]
    steps = 0

    def addToTree(oldn, newn):
        newn.parent = oldn
        newn.cost = oldn.cost + newn.distance(oldn)
        oldn.child = newn
        tree.append(newn)

        visual.drawEdge(oldn, newn, color='g', linewidth=1)
        visual.show()

    def getNeighbors(node, r):
        neighbors = []
        for otherNode in tree:
            if otherNode.distance(node) <= r:
                neighbors.append(otherNode)

        return neighbors
    
    def updateChildrenCosts(node):
        child = node.child
        if child is not None:
            child.cost = node.cost + node.distance(child)
            updateChildrenCosts(node.child)
    
    def rewire(newn, neighbors):
        for neighbor in neighbors:
            if neighbor is newn or neighbor is start:
                continue
            if neighbor.t <= newn.t:
                continue
            if not neighbor.connectsTo(newn):
                continue

            newCost = newn.cost + newn.distance(neighbor)

            if newCost < neighbor.cost:
                if neighbor.parent is not None:
                    visual.drawEdge(neighbor.parent, neighbor, color="yellow", linewidth=1)
                neighbor.parent = newn
                neighbor.cost = newCost
                visual.drawEdge(newn, neighbor, color='g', linewidth=1)
                updateChildrenCosts(neighbor)
                visual.show()

    bestGoal = None
    pathsFound = 0

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

        if not newn.inFreespace():
            continue

        n = len(tree) + 1
        radius = max(DSTEP * 2.0, 1.5 * DSTEP * sqrt(np.log(n)/n))
        neighbors = getNeighbors(newn, radius)

        validParents = []
        for neighbor in neighbors:
            if newn.t > neighbor.t and newn.connectsTo(neighbor):
                validParents.append(neighbor)
        if nearest not in validParents and newn.t > nearest.t and newn.connectsTo(nearest):
            validParents.append(nearest)
        if len(validParents) == 0:
            continue

        bestParent = min(validParents, key=lambda node: node.cost + node.distance(newn))
        addToTree(bestParent, newn)
        rewire(newn, neighbors)

        if newn.distance(goal) < DSTEP:
            curGoal = Node(goal.x, goal.y, newn.t + DT, map)
            if curGoal.connectsTo(newn):
                curGoal.parent = newn
                curGoal.cost = newn.cost + newn.distance(curGoal)

                if bestGoal is None or curGoal.cost < bestGoal.cost:
                    bestGoal = curGoal
                    pathsFound += 1
                    print("path found")
                    if pathsFound >= 4:
                        print("Exiting early since multiple paths found.")
                        break

        if len(tree) > 10 * ((map.xmax - map.xmin) * (map.ymax - map.ymin)) / (DSTEP ** 2):
            print("Exiting early due to size of tree.")
            break
    
    if bestGoal is not None:
        return build_path(bestGoal), tree

    return None, tree

# Kinodynamic Temporal RRT
def kinodynamicrrt(start, goal, visual, startAng=0):
    map = visual.map

    tree = [start]
    start.ang = startAng
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

        # choose new acceleration, velocity, and direction

        # newAcc = nearest.acc + random.uniform(-D_ACC_MAX, D_ACC_MAX)
        # newAcc = max(-ACC_MAX, min(ACC_MAX, newAcc))

        # newVel = nearest.vel + DT*newAcc
        # newVel = max(-VEL_MAX, min(VEL_MAX, newVel))
        # newAcc = (newVel - nearest.vel) / DT

        velDes = random.uniform(0.7 * VEL_MAX, VEL_MAX)
        accDes = (velDes - nearest.vel) / DT
        accDes = max(-ACC_MAX, min(ACC_MAX, accDes))
        dAcc = accDes - nearest.acc + random.uniform(-D_ACC_MAX * 0.4, D_ACC_MAX * 0.4)
        dAcc = max(-D_ACC_MAX, min(D_ACC_MAX, dAcc))
        newAcc = nearest.acc + dAcc
        newAcc = max(-ACC_MAX, min(ACC_MAX, newAcc))
        newVel = nearest.vel + DT * newAcc 
        newVel = max(0.0, min(VEL_MAX, newVel))
    
        newAng = (nearest.ang + random.uniform(-D_ANG_MAX, D_ANG_MAX)) % (2 * np.pi)
        
        newn = Node(nearest.x + DSTEP*DT*newVel*cos(newAng),
                    nearest.y + DSTEP*DT*newVel*sin(newAng),
                    nearest.t + DT,
                    map)
        newn.acc = newAcc
        newn.vel = newVel
        newn.ang = newAng

        if newn.inFreespace() and newn.connectsTo(nearest):
            addToTree(nearest, newn)

            if newn.distance(goal) < DSTEP:
                goal_reached = Node(goal.x, goal.y, newn.t + DT, map)
                if goal_reached.inFreespace() and goal_reached.connectsTo(newn):
                    goal_reached.parent = newn
                    return build_path(goal_reached), tree

    return None, tree

def kinodynamicrrtModified(start, goal, visual):
    map = visual.map

    tree = []
    numStartAngs = 8
    for i in range(numStartAngs):
        ang = 2*np.pi*i/numStartAngs
        s = Node(start.x, start.y, start.t, map)
        s.ang = ang
        tree.append(s)
    steps = 0

    def addToTree(oldn, newn):
        newn.parent = oldn
        tree.append(newn)

        visual.drawEdge(oldn, newn, color='g', linewidth=1)
        visual.show()

    def scoreClearance(node):
        return node.numCloseWalls(radius=0.3)

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

        velDes = random.uniform(0.7 * VEL_MAX, VEL_MAX)
        accDes = (velDes - nearest.vel) / DT
        accDes = max(-ACC_MAX, min(ACC_MAX, accDes))
        dAcc = accDes - nearest.acc + random.uniform(-D_ACC_MAX * 0.4, D_ACC_MAX * 0.4)
        dAcc = max(-D_ACC_MAX, min(D_ACC_MAX, dAcc))
        newAcc = nearest.acc + dAcc
        newAcc = max(-ACC_MAX, min(ACC_MAX, newAcc))
        newVel = nearest.vel + DT * newAcc 
        newVel = max(0.0, min(VEL_MAX, newVel))

        # Select the target angle to be angle with the best clearance score
        angleScores = []
        numCheck = 12

        for i in range(numCheck):
            checkOffset = -D_ANG_MAX + 2 * D_ANG_MAX * i / (numCheck - 1)
            candAng = (nearest.ang + checkOffset) % (2 * np.pi)

            checkX = nearest.x + DSTEP * DT * newVel * cos(candAng)
            checkY = nearest.y + DSTEP * DT * newVel * sin(candAng)
            checkNode = Node(checkX, checkY, nearest.t + DT, map)

            if not checkNode.inFreespace():
                continue
            if not checkNode.connectsTo(nearest):
                continue

            score = 2.5 * np.sqrt(checkNode.distance(target)) + checkNode.numCloseWalls(radius=0.2)
            angleScores.append((score, candAng))

        if len(angleScores) == 0:
            continue
        
        if random.random() < 0.5:
            angDes = min(angleScores, key=lambda x: x[0])[1] + random.uniform(-D_ANG_MAX * 0.4, D_ANG_MAX * 0.4)
        else:
            angDes = nearest.ang + random.uniform(-D_ANG_MAX, D_ANG_MAX)

        def wrap_to_pi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        angErr = wrap_to_pi(angDes - nearest.ang)
        dAng = max(-D_ANG_MAX, min(D_ANG_MAX, angErr))
        newAng = (nearest.ang + dAng) % (2 * np.pi)

        newn = Node(nearest.x + DSTEP*DT*newVel*cos(newAng),
                    nearest.y + DSTEP*DT*newVel*sin(newAng),
                    nearest.t + DT,
                    map)
        newn.acc = newAcc
        newn.vel = newVel
        newn.ang = newAng

        if newn.inFreespace() and newn.connectsTo(nearest):
            addToTree(nearest, newn)

            if newn.distance(goal) < DSTEP:
                goal_reached = Node(goal.x, goal.y, newn.t + DT, map)
                if goal_reached.inFreespace() and goal_reached.connectsTo(newn):
                    goal_reached.parent = newn
                    return build_path(goal_reached), tree

    return None, tree

def postProcess(path, enforceDynamics=False):
    shortpath = [path[0]]
    for i in range(2, len(path)):
        if not shortpath[-1].connectsTo(path[i], enforceDynamics=enforceDynamics):
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

    baseMaze = [
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
    maneuverMaze = [
        # Outer border
        H(0, 5, 5),
        H(0, 5, 0),
        V(0, 0, 5),
        V(5, 0, 5),

        # Inner walls
        H(0, 3, 1),
        H(2, 5, 2),
        H(0, 3, 3),
        H(2, 5, 4),
        HM(3, 5, 1),
        HM(0, 2, 2),
        HM(3, 5, 3),
        HM(0, 2, 4),
    ]
    deceptiveMaze = [
        # Outer border
        H(0, 5, 5),
        H(0, 5, 0),
        V(0, 0, 5),
        V(5, 0, 5),

        # Inner walls
        H(0, 3, 1),
        H(1, 4, 2),
        H(0, 3, 3),
        H(4, 5, 3),
        H(1, 4, 4),
        V(4, 1, 4)
    ]

    corridorMaze = [
        # Outer border
        H(0, 2, 5),
        H(0, 2, 0),
        V(0, 0, 5),
        V(2, 0, 5),

        # Inner walls
        HM(0, 2, 1),
        HM(0, 2, 2),
        HM(0, 2, 3),
        HM(0, 2, 4)
    ]
    
    map = Map(corridorMaze, XMIN, XMAX, YMIN, YMAX)

    visual = Visualization(map)

    start = Node(0.5, 0.5, 0.0, map)
    goal  = Node(1, 4.5, 0.0, map)

    visual.drawStartGoal(start, goal)
    visual.show("Showing basic world")

    if not start.inFreespace():
        print("Start is in collision at t=0. Adjust start.")
        return
    if not goal.inFreespace():
        print("Goal is in collision at t=0. Adjust goal.")
        return

    print("Planning...")
    path, tree = kinodynamicrrtModified(start, goal, visual)
    
    if path is None:
        print("Failed. Try increasing SMAX/NMAX or WAIT_PROB or gmax.")
        return
    
    print(f"Path found. tf={path[-1].t:.2f}, nodes={len(tree)}")

    # show final path before post-processing
    visual.draw_final_snapshot(path, tree, start, goal, color='blue')
    input("Final path before post-processing shown (with wall at final time).")

    #path = postProcess(path)
    path = postProcess(path, enforceDynamics=True)

    # show final path after post-processing
    visual.draw_final_snapshot(path, tree, start, goal, color='green')
    input("Final path after post-processing shown (with wall at final time). Press Enter to animate...")
    
    # show final animation
    plt.close(visual.fig)
    visual_ani = Visualization(map)
    ani = animation(path, visual_ani, start, goal)
    plt.show()

if __name__ == "__main__":
    main()