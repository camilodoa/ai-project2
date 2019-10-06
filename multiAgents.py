# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().data
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Our plan:
        # Winning > Not getting killed > eating food > moving closer to food > fearing ghosts (see: God)

        n = successorGameState.getNumFood()
        # If you can win that's the best possible move
        if n == 0:
            return 99999

        # Record food coordinates
        foods = []
        for i in range(len(newFood)):
            for j in range(len(newFood[i])):
                if newFood[i][j]:
                    foods.append((i, j))

        # Fear
        fear = 0
        fear_factor = 12
        ghosts = []
        gamma = .5

        if newGhostStates: # If there are ghosts
            for ghost in newGhostStates:
                if ghost.scaredTimer == 0:
                    md = manhattanDistance(ghost.getPosition(), newPos)
                    ghosts.append(md)

                    # If the spot gets us killed, it's an auto negative inf
                    if md == 0:
                        return -99999

        # Sort ghosts based on distance
        ghost = sorted(ghosts)

        for i in range(len(ghosts)):
            # Hunger is a negative exponential function that is multiplied by fear_factor
            fear += (fear_factor/ghosts[i]) * (gamma**i)


        hunger_factor = 18
        # Food hunger
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            return hunger_factor

        # Hunger factor
        hunger = 0
        if foods:
            closest_food = 99999
            for food in foods:
                md = manhattanDistance(food, newPos)
                if md == 0:
                    return hunger_factor
                # Loving the closest food most
                if md < closest_food:
                    closest_food = md

            for ghost in newGhostStates:
                if ghost.scaredTimer > 0:
                    md = manhattanDistance(ghost.getPosition(), newPos)
                    if md == 0:
                        return hunger_factor
                    # Loving the closest food most
                    if md < closest_food:
                        closest_food = md

            # Hunger is a negative exponential function that is multiplied by hunger_factor
            hunger = hunger_factor/closest_food

        score =  hunger - fear

        if score:
            # Moving is often better than sitting still
            if action == "Stop":
                return score - 5
            else:
                return score

        return successorGameState.getScore()


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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        
        max_score = 0
        max_action = None
        for action in gameState.getLegalActions(0):
          result = self.minimax(gameState.generateSuccessor(0, action), 0, 1)
          if result >= max_score:
            max_score = result
            max_action = action

        return max_action


        util.raiseNotDefined()

    def minimax(self, s, d, turn):
        '''
        s: gameState
        d: depth
        turn: 0 if pacman, 1 if ghost
        '''
        if s.isWin() or s.isLose():
            return self.evaluationFunction(s)

        if self.cutoffTest(d):
            return self.evaluationFunction(s)

        if turn == 0:
            max_action = 0
            actions = s.getLegalActions(0)

            for action in actions:
                result = self.minimax(s.generateSuccessor(0, action), d + 1, turn + 1)
                if result > max_action:
                    max_action = result

            return max_action


        if turn == 1:
            total_min = 0
            for i in range(1, s.getNumAgents()):
                min_action = 99999
                actions = s.getLegalActions(i)

                for action in actions:
                    result = self.minimax(s.generateSuccessor(i, action), d + 1, turn - 1)
                    if result < min_action:
                        min_action = result

                total_min += min_action
            return total_min



        # s.getLegalActions()



    def cutoffTest(self, d):
        if d >= self.depth:
            return True
        return False



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
