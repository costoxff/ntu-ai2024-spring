"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

Follow the project description for details.

Good luck and happy searching!
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    print("Solution:", [s, s, w, s, w, w, s, w])
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    frontier = util.Stack()
    visited = set()

    # (state, path)
    frontier.push((problem.getStartState(), []))

    while not frontier.isEmpty():
        cur_state, path = frontier.pop()

        if problem.isGoalState(cur_state):
            return path
        
        if cur_state in visited:
            continue
        
        visited.add(cur_state)
        for state, action, _ in problem.getSuccessors(cur_state):
            # append every traversal step in path
            # list.append with change the path of previous state, be careful.
            path_to_state = path + [action]
            frontier.push((state, path_to_state))

    return []
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # be careful node expand: https://stackoverflow.com/q/12765039

    frontier = util.Queue()
    visited = set()

    frontier.push((problem.getStartState(), []))

    while not frontier.isEmpty():
        cur_state, path = frontier.pop()

        if problem.isGoalState(cur_state):
            return path

        if cur_state in visited:
            continue

        visited.add(cur_state)
        for state, action, _ in problem.getSuccessors(cur_state):
            path_to_state = path + [action]
            frontier.push((state, path_to_state))

    return []
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    frontier = util.PriorityQueue()
    visited = set()

    # (state, path, cost)
    frontier.push((problem.getStartState(), [], 0), 0)

    while not frontier.isEmpty():
        cur_state, path, cur_cost = frontier.pop()

        if problem.isGoalState(cur_state):
            return path

        if cur_state in visited:
            continue

        visited.add(cur_state)
        for state, action, cost in problem.getSuccessors(cur_state):
            new_cost = cur_cost + cost
            path_to_state = path + [action]
            frontier.push((state, path_to_state, new_cost), new_cost)

    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # A star in 2 minite: https://youtu.be/71CEj4gKDnE?si=4JanYvZisIOhHRXe

    frontier = util.PriorityQueue()
    visited = set()

    start_state = problem.getStartState()
    # f(n) = g(n) + h(n):
    # g(n) is Dijkstra's algorithm, h(n) is any greedy algorithm
    frontier.push((start_state, [], 0), heuristic(start_state, problem))
    # frontier.push((start_state, [], 0), 0)

    while not frontier.isEmpty():
        cur_state, path, cur_cost = frontier.pop()

        if problem.isGoalState(cur_state):
            return path

        if cur_state in visited:
            continue
        
        visited.add(cur_state)
        for state, action, cost in problem.getSuccessors(cur_state):
            new_cost = cur_cost + cost
            path_to_state = path + [action]
            p = new_cost + heuristic(state, problem)
            # p = new_cost
            frontier.push((state, path_to_state, new_cost), p)

    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
