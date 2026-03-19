import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import random
import time
import multiprocessing as mp
from math import sqrt, atan2, cos, sin
from shapely.geometry import Point, LineString, Polygon

# Parameters
DSTEP = 0.5
DT = 0.4
SMAX = 50000
NMAX = 50000

GOAL_BIAS = 0.06
WAIT_PROB = 0.22
TIME_WEIGHT = 0.2

XMIN, XMAX = 0, 6
YMIN, YMAX = 0, 7
VMIN, VMAX = 0.0, 1.5

WALL_THICKNESS = 0.1
WALL_SPLIT_OMEGA = 0.28



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



# Splitting Wall (inherits Wall): One wall that becomes two segments
class SplitWall(Wall):
    def __init__(self, xmin, xmax, ymin, ymax, gappos, gapmax, omega, splitdir=0):
        super().__init__(xmin, xmax, ymin, ymax)
        self.gappos = gappos  # center of gap
        self.splitdir = splitdir  # 0: x, 1: y
        self.gapmax = gapmax
        self.omega = omega

    def gap(self, t):
        return self.gapmax * (np.sin(self.omega * t) ** 2)

    def polygonsAt(self, t):
        g = self.gap(t)
        polys = []

        if self.splitdir == 0:  # split in x direction
            xL0 = self.xmin
            xR1 = self.xmax

            xR0 = self.gappos - g / 2.0
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
        else:  # split in y direction
            yB0 = self.ymin
            yT1 = self.ymax

            yT0 = self.gappos - g / 2.0
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


class Map:
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

    def spatial_distance(self, other):
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance(self, other):
        return sqrt((self.x - other.x) ** 2 +
                    (self.y - other.y) ** 2 +
                    (TIME_WEIGHT * (self.t - other.t)) ** 2)

    def inFreespace(self):
        if self.x <= XMIN or self.x >= XMAX or self.y <= YMIN or self.y >= YMAX:
            return False
        p = Point(self.x, self.y)
        for poly in self.map.polygonsAt(self.t):
            if not poly.disjoint(p):
                return False
        return True

    def connectsTo(self, other):
        if other.t < self.t:
            return False
        line = LineString([(self.x, self.y), (other.x, other.y)])
        for poly in self.map.polygonsAt(other.t):
            if not poly.disjoint(line):
                return False
        return True



# Visualization Class
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
            patch = MplPolygon(
                list(zip(x, y)),
                closed=True,
                facecolor='black',
                edgecolor='black'
            )
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
        for i in range(len(path) - 1):
            self.drawEdge(path[i], path[i + 1], **kwargs)

    def drawStartGoal(self, start, goal):
        self.drawNode(start, color='orange', marker='o')
        self.drawNode(goal, color='purple', marker='o')

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

        for n in tree:
            if n.parent:
                self.drawEdge(n.parent, n, color='lightgreen', linewidth=0.5)

        for i in range(len(path) - 1):
            self.drawEdge(path[i], path[i + 1], color=color, linewidth=3)

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


class NoOpVisualization:
    def __init__(self, map):
        self.map = map

    def drawMap(self, t):
        pass

    def show(self, text=''):
        pass

    def drawNode(self, node, **kwargs):
        pass

    def drawEdge(self, head, tail, **kwargs):
        pass

    def drawPath(self, path, **kwargs):
        pass

    def drawStartGoal(self, start, goal):
        pass

    def drawRobot(self, x, y):
        pass

    def draw_final_snapshot(self, path, tree, start, goal, color='red'):
        pass

    def drawFrame(self, path, start, goal, t):
        pass

    def clear(self):
        pass


# Helpers
def build_path(goal):
    path = [goal]
    while path[0].parent:
        path.insert(0, path[0].parent)
    return path


def path_length(path):
    if path is None or len(path) < 2:
        return None
    total = 0.0
    for i in range(len(path) - 1):
        total += path[i].spatial_distance(path[i + 1])
    return total


# Temporal RRT without time
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
                          0.0, map)

        nearest = min(tree, key=lambda n: n.spatial_distance(target))

        if random.random() < WAIT_PROB:
            newn = Node(nearest.x, nearest.y, nearest.t + DT, map)
        else:
            ang = atan2(target.y - nearest.y, target.x - nearest.x)
            newn = Node(nearest.x + DSTEP * cos(ang),
                        nearest.y + DSTEP * sin(ang),
                        nearest.t + DT,
                        map)

        if newn.inFreespace() and nearest.connectsTo(newn):
            addToTree(nearest, newn)

            if newn.spatial_distance(goal) < DSTEP:
                goal_reached = Node(goal.x, goal.y, newn.t + DT, map)
                if goal_reached.inFreespace() and newn.connectsTo(goal_reached):
                    goal_reached.parent = newn
                    return build_path(goal_reached), tree

    return None, tree


# Temporal RRT with time
def rrt_time(start, goal, visual):
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

        t_max = (sqrt((goal.x - start.x) ** 2 + (goal.y - start.y) ** 2) / max(VMAX, 1e-6)) * 10

        if random.random() < GOAL_BIAS:
            t_latest = max(n.t for n in tree)
            target = Node(goal.x, goal.y, t_latest, map)
        else:
            target = Node(random.uniform(map.xmin, map.xmax),
                          random.uniform(map.ymin, map.ymax),
                          random.uniform(0.0, t_max),
                          map)

        nearest = min(tree, key=lambda n: n.distance(target))

        ang = atan2(target.y - nearest.y, target.x - nearest.x)
        v = random.uniform(VMIN, VMAX)
        step = v * DT
        newn = Node(nearest.x + step * cos(ang),
                    nearest.y + step * sin(ang),
                    nearest.t + DT,
                    map)

        if newn.inFreespace() and nearest.connectsTo(newn):
            addToTree(nearest, newn)

            if newn.spatial_distance(goal) < DSTEP:
                goal_reached = Node(goal.x, goal.y, newn.t + DT, map)
                if goal_reached.inFreespace() and newn.connectsTo(goal_reached):
                    goal_reached.parent = newn
                    return build_path(goal_reached), tree

    return None, tree


def postProcess(path):
    shortpath = [path[0]]
    for i in range(2, len(path)):
        if not shortpath[-1].connectsTo(path[i]):
            shortpath.append(path[i - 1])
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


# Maze builders
def H(x1, x2, y):
    return Wall(min(x1, x2), max(x1, x2), y - WALL_THICKNESS / 2, y + WALL_THICKNESS / 2)


def V(x, y1, y2):
    return Wall(x - WALL_THICKNESS / 2, x + WALL_THICKNESS / 2, min(y1, y2), max(y1, y2))


def HM(x1, x2, y):
    return SplitWall(min(x1, x2), max(x1, x2), y - WALL_THICKNESS / 2, y + WALL_THICKNESS / 2,
                     gappos=(x1 + x2) / 2, gapmax=1, omega=WALL_SPLIT_OMEGA)


def VM(x, y1, y2):
    return SplitWall(x - WALL_THICKNESS / 2, x + WALL_THICKNESS / 2, min(y1, y2), max(y1, y2),
                     gappos=(y1 + y2) / 2, gapmax=1, omega=WALL_SPLIT_OMEGA, splitdir=1)


def get_base_maze():
    return [
        H(0, 6, 7),
        H(0, 6, 0),
        V(0, 0, 7),
        V(6, 0, 7),

        H(1, 5, 6),
        H(1, 2, 5),
        H(3, 5, 5),
        H(2, 3, 4),
        HM(4, 6, 4),
        H(0, 1, 3),
        H(2, 5, 3),
        H(0, 4, 2),
        HM(2, 3, 1),
        H(4, 5, 1),
        V(1, 0, 1),
        V(1, 3, 5),
        VM(2, 4, 5),
        V(3, 1, 2),
        VM(4, 0, 1),
        V(4, 4, 5),
        VM(5, 1, 3),
        V(5, 5, 6)
    ]


def get_maneuver_maze():
    return [
        H(0, 5, 5),
        H(0, 5, 0),
        V(0, 0, 5),
        V(5, 0, 5),

        H(0, 3, 1),
        H(2, 5, 2),
        H(0, 3, 3),
        H(2, 5, 4),
        HM(3, 5, 1),
        HM(0, 2, 2),
        HM(3, 5, 3),
        HM(0, 2, 4),
    ]


def get_deceptive_maze():
    return [
        H(0, 5, 5),
        H(0, 5, 0),
        V(0, 0, 5),
        V(5, 0, 5),

        H(0, 3, 1),
        H(1, 4, 2),
        H(0, 3, 3),
        H(4, 5, 3),
        H(1, 4, 4),
        V(4, 1, 4)
    ]


def get_corridor_maze():
    return [
        H(0, 2, 5),
        H(0, 2, 0),
        V(0, 0, 5),
        V(2, 0, 5),

        HM(0, 2, 1),
        HM(0, 2, 2),
        HM(0, 2, 3),
        HM(0, 2, 4)
    ]


def build_benchmark_problem(maze_name):
    if maze_name == "corridor":
        map_obj = Map(get_corridor_maze(), XMIN, XMAX, YMIN, YMAX)
        start = Node(0.5, 0.5, 0.0, map_obj)
        goal = Node(1.5, 4.5, 0.0, map_obj)
        return map_obj, start, goal

    if maze_name == "deceptive":
        map_obj = Map(get_deceptive_maze(), XMIN, XMAX, YMIN, YMAX)
        start = Node(0.5, 0.5, 0.0, map_obj)
        goal = Node(4.5, 4.5, 0.0, map_obj)
        return map_obj, start, goal

    if maze_name == "maneuver":
        map_obj = Map(get_maneuver_maze(), XMIN, XMAX, YMIN, YMAX)
        start = Node(0.5, 0.5, 0.0, map_obj)
        goal = Node(4.5, 4.5, 0.0, map_obj)
        return map_obj, start, goal

    if maze_name == "base":
        map_obj = Map(get_base_maze(), XMIN, XMAX, YMIN, YMAX)
        start = Node(0.5, 0.5, 0.0, map_obj)
        goal = Node(4.5, 4.5, 0.0, map_obj)
        return map_obj, start, goal
    raise ValueError(f"Unknown maze name: {maze_name}")


def planner_dispatch(planner_name, start, goal, visual):
    if planner_name == "rrt":
        return rrt(start, goal, visual)
    if planner_name == "rrt_time":
        return rrt_time(start, goal, visual)
    raise ValueError(f"Unknown planner: {planner_name}")


# Benchmark
def benchmark_worker(planner_name, maze_name, queue):
    try:
        map_obj, start, goal = build_benchmark_problem(maze_name)
        visual = NoOpVisualization(map_obj)

        t0 = time.perf_counter()
        path, tree = planner_dispatch(planner_name, start, goal, visual)
        elapsed = time.perf_counter() - t0

        queue.put({
            "success": path is not None,
            "compute_time": elapsed,
            "path_length": path_length(path) if path is not None else None,
            "final_time": path[-1].t if path is not None else None,
            "total_nodes": len(tree) if tree is not None else None
        })
    except Exception as e:
        queue.put({
            "success": False,
            "compute_time": None,
            "path_length": None,
            "final_time": None,
            "total_nodes": None,
            "error": str(e)
        })


def run_single_benchmark(planner_name, maze_name, run_idx, timeout_sec=30.0):
    queue = mp.Queue()

    p = mp.Process(target=benchmark_worker, args=(planner_name, maze_name, queue))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "success": False,
            "timeout": True,
            "compute_time": timeout_sec,
            "path_length": None,
            "final_time": None,
            "total_nodes": None
        }

    if not queue.empty():
        result = queue.get()
        result["timeout"] = False
        return result

    return {
        "success": False,
        "timeout": False,
        "compute_time": None,
        "path_length": None,
        "final_time": None,
        "total_nodes": None
    }


def safe_mean(values):
    vals = [v for v in values if v is not None]
    if len(vals) == 0:
        return None
    return sum(vals) / len(vals)


def benchmark_planner(planner_name, maze_name, trials=50, timeout_sec=120.0):
    results = []

    print(f"\nBenchmarking {planner_name} on {maze_name} maze")
    print("-" * 72)

    for i in range(trials):
        result = run_single_benchmark(planner_name, maze_name, i, timeout_sec)
        results.append(result)

        if result["timeout"]:
            status = "TIMEOUT"
        elif result["success"]:
            status = "SUCCESS"
        else:
            status = "FAIL"

        print(f"Run {i + 1:02d}/{trials}: {status:7s} | "
              f"time = {result['compute_time'] if result['compute_time'] is not None else float('nan'):.4f}s")

    success_runs = [r for r in results if r["success"]]

    success_rate = 100.0 * len(success_runs) / trials
    avg_compute_time = safe_mean([r["compute_time"] for r in results])
    avg_path_length = safe_mean([r["path_length"] for r in success_runs])
    avg_final_time = safe_mean([r["final_time"] for r in success_runs])
    avg_total_nodes = safe_mean([r["total_nodes"] for r in results])

    print("\nSummary")
    print("-" * 72)
    print(f"Success rate:              {success_rate:.2f}%")
    if avg_compute_time is None:
        print("Average computing time:    N/A")
    else:
        print(f"Average computing time:    {avg_compute_time:.4f} s")
    if avg_path_length is None:
        print("Average path length:       N/A")
    else:
        print(f"Average path length:       {avg_path_length:.4f}")
    if avg_final_time is None:
        print("Average path final time:   N/A")
    else:
        print(f"Average path final time:   {avg_final_time:.4f}")
    if avg_total_nodes is None:
        print("Average total node sampled:N/A")
    else:
        print(f"Average total node sampled:{avg_total_nodes:.2f}")

    return {
        "planner": planner_name,
        "maze": maze_name,
        "success_rate": success_rate,
        "avg_compute_time": avg_compute_time,
        "avg_path_length": avg_path_length,
        "avg_final_time": avg_final_time,
        "avg_total_nodes": avg_total_nodes,
        "results": results
    }


def benchmark_all():
    planners = ["rrt", "rrt_time"]
    mazes = ["base", "maneuver"]

    all_stats = []
    for maze_name in mazes:
        for planner_name in planners:
            stats = benchmark_planner(planner_name, maze_name, trials=30, timeout_sec=30.0)
            all_stats.append(stats)

    print("\n" + "=" * 96)
    print("FINAL BENCHMARK TABLE")
    print("=" * 96)
    print(f"{'Planner':<12}{'Maze':<12}{'Success Rate':<16}{'Avg Time (s)':<16}"
          f"{'Avg Path Len':<16}{'Avg Final T':<16}{'Avg Nodes':<12}")
    print("-" * 96)

    for s in all_stats:
        avg_time = "N/A" if s["avg_compute_time"] is None else f"{s['avg_compute_time']:.4f}"
        avg_len = "N/A" if s["avg_path_length"] is None else f"{s['avg_path_length']:.4f}"
        avg_tf = "N/A" if s["avg_final_time"] is None else f"{s['avg_final_time']:.4f}"
        avg_nodes = "N/A" if s["avg_total_nodes"] is None else f"{s['avg_total_nodes']:.2f}"

        print(f"{s['planner']:<12}{s['maze']:<12}{s['success_rate']:.2f}%"f"{avg_time:<16}{avg_len:<16}{avg_tf:<16}{avg_nodes:<12}")

    return all_stats


#main
def main():
    map = Map(get_base_maze(), XMIN, XMAX, YMIN, YMAX)

    visual = Visualization(map)

    start = Node(0.5, 0.5, 0.0, map)
    goal = Node(4.5, 4.5, 0.0, map)

    visual.drawStartGoal(start, goal)
    visual.show("Showing basic world")

    if not start.inFreespace():
        print("Start is in collision at t=0. Adjust start.")
        return
    if not goal.inFreespace():
        print("Goal is in collision at t=0. Adjust goal.")
        return

    print("Planning...")
    #path, tree = rrt(start, goal, visual)
    path, tree = rrt_time(start, goal, visual)

    if path is None:
        print("Failed. Try increasing SMAX/NMAX or WAIT_PROB or gmax.")
        return

    print(f"Path found. tf={path[-1].t:.2f}, nodes={len(tree)}")

    visual.draw_final_snapshot(path, tree, start, goal, color='blue')
    input("Final path before post-processing shown (with wall at final time).")

    visual.draw_final_snapshot(path, tree, start, goal, color='green')
    input("Final path after post-processing shown (with wall at final time). Press Enter to animate...")

    plt.close(visual.fig)
    visual_ani = Visualization(map)
    ani = animation(path, visual_ani, start, goal)
    plt.show()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    #main()
    #benchmark_planner("rrt", "corridor")
    #benchmark_planner("rrt_time", "deceptive")
    benchmark_all()