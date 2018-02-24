# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util #contains data structures

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).

  You do not need to change anything in this class, ever.
  """

  def getStartState(self):
     """
     Returns the start state for the search problem
     """
     util.raiseNotDefined()

  def isGoalState(self, state):
     """
       state: Search state

     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state

     For a given state, this should return a list of triples,
     (successor, action, stepCost), where 'successor' is a
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take

     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()


def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def graphSearch(problem, strategy):
    """
    PARAMETERS:
    	problem: initializes board state
    	strategy: type of ordering structure for search
    
    graphSearch() takes a pacman problem and a strategy (data structure) to
    find a path from the start state to the goal state. graphSearch() is an
    uninformed search method. It expands out possible plans, maintains a frontier
    of unexpanded search nodes, and tries to expand as few nodes as possible.

    Returns a sequence of moves that solves the given problem.
    """
    start_node = Node(problem.getStartState(), None , 0, None) # creates a node from the starting state
    strategy.push(start_node)
    explored = set() # creates empty set for explored nodes

    # searches until the frontier is empty
    while strategy.isEmpty() == False: 
        node = strategy.pop() # pops the next node from stack
        if node.state in explored: continue 
        if problem.isGoalState(node.state) == True:
            return node.getSolution()
        else:
            explored.add(node.state)

	    # expand chosen node, adding the resulting nodes to the frontier if they have not been explored 
            for succ in problem.getSuccessors(node.state):
                succ_node = Node(succ[0],succ[1],succ[2], node)
                if succ_node.state not in explored and succ_node not in strategy.list:
                    strategy.push(succ_node)


def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].

  uses graphSearch with a stack (LIFO) strategy
  """
  return graphSearch(problem, util.Stack())

def breadthFirstSearch(problem):
  """
  Search the shallowest nodes in the search tree first. [p 81]

  uses graphSearch with a queue (FIFO) strategy
  """
  return graphSearch(problem, util.Queue())

def uniformCostSearch(problem):
  """
  Search the node of least total cost first.

  operates similar to graphSearch, but with
  extra cost checking functionality

  uniformCostSearch() takes a pacman problem and tries to
  find an optimal and complete path from the start state to the goal state. uniformCostSearch()
  is an uninformed search method. It expands out possible plans, maintains a frontier
  of unexpanded search nodes, and tries to expand as few nodes as possible.

  Returns a sequence of moves that solves the given problem. 
  """
  strategy = util.PriorityQueue()
  start_node = Node(problem.getStartState(), None , 0, None) # creates a node from the starting state
  strategy.push(start_node, start_node.cost)
  explored = set() # creates empty set for explored nodes

  # searches until the frontier is empty
  while strategy.isEmpty() == False:
      node = strategy.pop() # pops the next node from the priority queue
      if node.state in explored: continue
      if problem.isGoalState(node.state) == True:
          return node.getSolution()
      else:
          explored.add(node.state)
      
   	  # expand the chosen node 
          for succ in problem.getSuccessors(node.state):
              succ_node = Node(succ[0],succ[1],node.cost+ succ[2], node)
	      # add node to the frontier if they have not been explored or are not already in priority queue
              if succ_node.state not in explored and succ_node not in strategy.heap:
                  strategy.push(succ_node, succ_node.cost)
	      # add the node with the lest total cost to the frontier
              elif succ_node in strategy.heap:
                  compare_node = strategy.find_and_extract(succ_node.state)
                  if succ_node.cost < compare_node.cost:
                      strategy.push(succ_node, succ_node.cost)
                  else:
                      strategy.push(compare_node, compare_node.cost)


def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first.

    Using an admissible heuristic to gaurantee that the first solution found
    will be an optimal one. Expands nodes and adds them to the frontier as long
    as they have not been previously explored. Uses lowest combined cost and
    the specified heuristic function to find optimal path. 
  
    PARAMETERS: 
	problem: gives the current pacman problem(state, map, food, walls)
	heuristic: the specific heuristic used for this instance of search

    Returns a path from the start state to the goal state.
    """
    heur_start = heuristic(problem.getStartState(), problem) # creates the heuristic object
    strategy = util.PriorityQueue()
    start_node = Node(problem.getStartState(), None , 0, None)
    strategy.push(start_node, heuristic(start_node.state, problem) +start_node.cost)
    explored = set() # creates an empty set for the previously expanded nodes

    # iterates until frontier is empty
    while strategy.isEmpty() == False:
        node = strategy.pop()
        if node.state in explored: continue
        if problem.isGoalState(node.state) == True:
          return node.getSolution()
        else: # if the node is not the goal state, try to add nodes to frontier
          explored.add(node.state)
          for succ in problem.getSuccessors(node.state):
              succ_node = Node(succ[0],succ[1],node.cost+ succ[2], node)
	      # if succ_node is not in explored or in the frontier, add to frontier
              if succ_node.state not in explored and succ_node not in strategy.heap:
                  strategy.push(succ_node, succ_node.cost + heuristic(succ_node.state, problem))

	      # if succ_node is in frontier, add node that had cheaper path and heuristic to frontier
              elif succ_node in strategy.heap:
                  compare_node = strategy.find_and_extract(succ_node.state)
                  if succ_node.cost < compare_node.cost:
                      strategy.push(succ_node, succ_node.cost + heuristic(succ_node.state, problem))
                  else:
                      strategy.push(compare_node, compare_node.cost + heuristic(compare_node.state, problem))




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

#shorthand directons
from game import Directions
s = Directions.SOUTH
w = Directions.WEST
n = Directions.NORTH
e = Directions.EAST

class Node():
    """
    Node class is a full representation of a node in a search graph, with attributes:

    *state encodes all the necessary components for the problem to understand the world (subset of Pacman gameState)
    *action is the move necessary to reach the node from its parent
    *cost is the full cost of the path from start to the node
    *parent node necessary to traverse back up the tree to get a solution path
    """

    def __init__(self, state, action, cost, parent):

        self.state = state
        if(action == 'North'):
            self.action = n
        elif(action == 'South'):
            self.action = s
        elif(action == 'West'):
            self.action = w
        elif(action =='East'):
            self.action = e
        else:
            self.action = None
        self.cost = cost
        self.parent = parent


    def getSolution(self):
        """
	This function returns the path to the starting state of the problem.
        It does so by adding the action each parent took to get to the current node
	and adding it to the list. Finally the reversed list is returned.

        """
        path = [self.action]
        parent = self.parent
        while parent.action != None:
            path.append(parent.action)
            parent = parent.parent
        return path[::-1]
