# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import math
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    ghost_pos = [ghostState.getPosition() for ghostState in newGhostStates]
    food_dist = []

    for i in oldFood.asList():
        food_dist.append(manhattanDistance(newPos, i))
        min_food = min(food_dist)

    if oldFood[newPos[0]][newPos[1]] == True: x = 20
    else: x= 0

    min_ghost_dist = min([manhattanDistance(ghost_pos[x], newPos) for x in xrange(len(ghost_pos))])
    return   x - math.exp(-(min_ghost_dist-5)) - 1.5*min_food

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """


  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    actions_list = []
    for a in gameState.getLegalActions(0):

        action_val, ply = self.min_value(gameState.generateSuccessor(0, a), 0, 1)
        actions_list.append((action_val, a))
        take_action = max(actions_list)
    #print(gameState.getNumAgents())
    print(take_action[0])
    return take_action[1]

  def min_value(self, gameState, ply, agent):
      if gameState.isWin() or gameState.isLose() or ply == self.depth*gameState.getNumAgents()-1:
          return (self.evaluationFunction(gameState), ply)

      v = 1000000000000000000000000
      for action in gameState.getLegalActions(agent):
          if agent < gameState.getNumAgents()-1:
              test_action, new_ply = self.min_value(gameState.generateSuccessor(agent,action), ply +1, agent+1)
              v = min(v, test_action)
          else:
              test_action, new_ply = self.max_value(gameState.generateSuccessor(agent,action), ply +1, 0)
              v = min(v, test_action)

      return (v, new_ply)

  def max_value(self, gameState, ply, agent):
      if gameState.isWin() or gameState.isLose() or ply == self.depth*gameState.getNumAgents()-1: return self.evaluationFunction(gameState), ply
      util_val = -1000000000000000000000
      for a in gameState.getLegalActions(agent):
          test_action, new_ply = self.min_value(gameState.generateSuccessor(agent, a), ply +1, agent +1 )
          util_val = max(util_val, test_action)

      return (util_val, new_ply)


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    actions_list = []
    #print gameState.getLegalActions(0)\
    best_action = Directions.STOP
    for a in gameState.getLegalActions(0):

        action_val = self.min_value_ab(gameState.generateSuccessor(0, a), 0, 1, -10000000,10000000)
        actions_list.append((action_val, a))
        take_action = max(actions_list)
        best_action = take_action[1]

    #print(gameState.getNumAgents())
    print(take_action[0])

    return best_action
    #return self.max_value_ab(gameState.generateSuccessor(0, a), 0, 1, -10000000,10000000)

  def min_value_ab(self, gameState, ply, agent, alpha, beta):
    #  print alpha
    #  print beta
      if gameState.isWin() or gameState.isLose() or ply == (self.depth*gameState.getNumAgents()): return (self.evaluationFunction(gameState))

      v = 10000000

      for a in gameState.getLegalActions(agent):
          if agent < gameState.getNumAgents()-1:
              test_action = self.min_value_ab(gameState.generateSuccessor(agent,a), ply +1, agent+1, alpha, beta)
              v = min(v, test_action)
          else:
              test_action = self.max_value_ab(gameState.generateSuccessor(agent,a), ply +1, 0, alpha, beta)
              #print test_action
              v = min(v, test_action)
          #print v
          if v <= alpha: return v

          beta = min(beta, v)
      return v

  def max_value_ab(self, gameState, ply, agent, alpha, beta):
      if gameState.isWin() or gameState.isLose() or (ply == self.depth*gameState.getNumAgents()-1): return (self.evaluationFunction(gameState))

      v = -10000000
      for a in gameState.getLegalActions(agent):

          test_action = self.min_value_ab(gameState.generateSuccessor(agent, a), ply +1, agent +1 , alpha, beta)
          v = max(v, test_action)
          if v >= beta: return v

          alpha = max(alpha, v)
      return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
    Returns the expectimax action using self.depth and self.evaluationFunction

    All ghosts should be modeled as choosing uniformly at random from their
    legal moves.
        """
        actions_list = []
        for a in gameState.getLegalActions(0):
            action_val, ply = self.chance_val(gameState.generateSuccessor(0, a), 0, 1)
            actions_list.append((action_val, a))
            take_action = max(actions_list)

        return take_action[1]

    def max_value(self, gameState, ply, agent):
        if gameState.isWin() or gameState.isLose() or ply == self.depth*gameState.getNumAgents()-1: return self.evaluationFunction(gameState), ply
        util_val = -1000000000000000000000
        for a in gameState.getLegalActions(agent):
            test_action, new_ply = self.chance_val(gameState.generateSuccessor(agent, a), ply +1, agent +1 )
            util_val = max(util_val, test_action)

        return (util_val, new_ply)

    def chance_val(self, gameState, ply, agent):
        if gameState.isWin() or gameState.isLose() or ply == self.depth*gameState.getNumAgents()-1: return self.evaluationFunction(gameState), ply
        p_move = 1/len(gameState.getLegalActions(agent))
        v = 0
        for a in gameState.getLegalActions(agent):
            if agent < gameState.getNumAgents()-1:
                test_action, new_ply = self.chance_val(gameState.generateSuccessor(agent,a), ply +1, agent+1)
                v += p_move * test_action
            else:
                test_action, new_ply = self.max_value(gameState.generateSuccessor(agent,a), ply +1, 0)
                v += p_move * test_action

        return (v, new_ply)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """

  pos = currentGameState.getPacmanPosition()
  oldFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  ghost_pos = [ghostState.getPosition() for ghostState in newGhostStates]
  food_dist = []

  for i in oldFood.asList():
      food_dist.append(manhattanDistance(pos, i))
      min_food = min(food_dist)


  return_val = 0
  min_ghost_dist = min([manhattanDistance(ghost_pos[x], pos) for x in xrange(len(ghost_pos))])
  if min_ghost_dist  < 3 : return_val += -400

  if min_food < min_ghost_dist: return_val += 100
  if oldFood[pos+1][pos] ==True or oldFood[pos-1][pos] ==True or oldFood[pos-1][pos]==True  or oldFood[pos][pos-1] ==True: return_val +=300
  return  return_val


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
