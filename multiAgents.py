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
        ghosts = sorted(ghosts)

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

        max_score = -99999
        max_action = None
        for action in gameState.getLegalActions(0):
          result = self.minimax(gameState.generateSuccessor(0, action), 1, 1)
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
            return self.evaluationFunction(s) # FIX THIS

        if turn == 0:
            max_action = -99999
            actions = s.getLegalActions(0)

            for action in actions:
                result = self.minimax(s.generateSuccessor(0, action), d, turn + 1)
                if result > max_action:
                    max_action = result

            return max_action


        if turn >= 1:
            min_action = 99999
            actions = s.getLegalActions(turn)

            for action in actions:
                if turn == s.getNumAgents()-1:
                    result = self.minimax(s.generateSuccessor(turn, action), d + 1, 0)
                else:
                    result = self.minimax(s.generateSuccessor(turn, action), d, turn + 1)

                if result < min_action:
                    min_action = result

            return min_action


    def cutoffTest(self, d):
        if d > self.depth:
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
        max_score = -99999
        max_action = None
        alpha = -99999
        beta = 99999

        for action in gameState.getLegalActions(0):
          result = self.minimax(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)

          if result >= max_score:
            max_score = result
            max_action = action

          # Pruning part
          if max_score > beta:
              return max_action

          alpha = max(alpha, max_score)

        return max_action

        util.raiseNotDefined()

    def minimax(self, s, d, turn, alpha, beta):
        '''
        s: gameState
        d: depth
        turn: 0 if pacman, 1 if ghost
        '''
        if s.isWin() or s.isLose():
            return self.evaluationFunction(s)

        if self.cutoffTest(d):
            return self.evaluationFunction(s) # FIX THIS

        if turn == 0:
            max_action = -99999
            actions = s.getLegalActions(0)

            for action in actions:
                result = self.minimax(s.generateSuccessor(0, action), d, turn + 1, alpha, beta)

                if result > max_action:
                    max_action = result

                # Pruning part
                if max_action > beta:
                    return max_action

                alpha = max(alpha, max_action)

            return max_action


        if turn >= 1:
            min_action = 99999
            actions = s.getLegalActions(turn)

            for action in actions:
                if turn == s.getNumAgents()-1:
                    result = self.minimax(s.generateSuccessor(turn, action), d + 1, 0, alpha, beta)
                else:
                    result = self.minimax(s.generateSuccessor(turn, action), d, turn + 1, alpha, beta)

                if result < min_action:
                    min_action = result

                # Pruning part
                if min_action < alpha:
                    return min_action

                beta = min(beta, min_action)

            return min_action


    def cutoffTest(self, d):
        if d > self.depth:
            return True
        return False

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
        max_score = -99999
        max_action = None
        for action in gameState.getLegalActions(0):
          result = self.minimax(gameState.generateSuccessor(0, action), 1, 1)
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
            return self.evaluationFunction(s) # FIX THIS

        if turn == 0:
            max_action = -99999
            actions = s.getLegalActions(0)

            for action in actions:
                result = self.minimax(s.generateSuccessor(0, action), d, turn + 1)
                if result > max_action:
                    max_action = result

            return max_action


        if turn >= 1:
            sum_action = 0.0
            actions = s.getLegalActions(turn)

            for action in actions:
                if turn == s.getNumAgents()-1:
                    result = self.minimax(s.generateSuccessor(turn, action), d + 1, 0)
                else:
                    result = self.minimax(s.generateSuccessor(turn, action), d, turn + 1)

                sum_action += result

            return sum_action / len(actions)


    def cutoffTest(self, d):
        if d > self.depth:
            return True
        return False

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <We calculate a score based on a linear combination of a few features. 
      This linear combination is generated by adding "hunger", which is derived by summing the reciprocal distances from all the available food, each with a discount factor of -0.4 to the nth power; 
      subtracting "fear", which is calculated the same as hunger, but is based on the distances to ghosts (who are not scared), multiplied by a discount factor of 0.5 to the nth power; 
      adding a random value from a very small unifrom distribution (0 to 0.5), to break ties arbitrarily; 
      subtracting the amount of food left, squared; adding the score of the game; 
      and subtracting the number of powerup capsules left, squared. 
      Our function also makes pacman chase after ghosts who are scared, by incorporating scared ghosts into the hunger factor with a greater weight than food. 
      Capsules are also favored over regular food when there's little food left.>
    """
    # Our plan:
    # Winning > Not getting killed > eating food > moving closer to food > fearing ghosts (see: God)
    ghostStates = currentGameState.getGhostStates()
    n = currentGameState.getNumFood()
    pos = currentGameState.getPacmanPosition()
    foodStates = currentGameState.getFood().data
    capsules = currentGameState.getCapsules()

    # If you can win that's the best possible move
    if currentGameState.isWin():
        return 99999 + random.uniform(0, .5)

    if currentGameState.isLose():
        return -99999

    # Fear
    fear = 0
    fear_factor = 10
    ghosts = []
    gamma = .5

    # Calculate distances to nearest ghost
    if ghostStates:
        for ghost in ghostStates:
            if ghost.scaredTimer == 0:
                md = manhattanDistance(ghost.getPosition(), pos)
                ghosts.append(md)


    # Sort ghosts based on distance
    ghosts = sorted(ghosts)
    # Only worry about ghosts if they're nearby
    ghosts = [ghost for ghost in ghosts if ghost < 5]


    for i in range(len(ghosts)):
        # Fear is sum of the recipricals of the distances to the nearest ghosts multiplied
        # by a gamma^i where 0<gamma<1 and by a fear_factor
        fear += (fear_factor/ghosts[i]) * (gamma**i)

    # Record food coordinates
    foods = []
    for i in range(len(foodStates)):
        for j in range(len(foodStates[i])):
            if foodStates[i][j]:
                foods.append((i, j))

    #Calculate distances to nearest foods
    foodDistances = []
    if foods:
        for food in foods:
            md = manhattanDistance(food, pos)
            foodDistances.append(md)
    foodDistances = sorted(foodDistances)


    hunger_factor = 18
    # Hunger factor
    hunger = 0
    foodGamma = -0.4
    for i in range(len(foodDistances)):
        # Hunger is the sum of the reciprical of the distances to the nearest foods multiplied
        # by a foodGamma^i where 0<foodGamma<1 and by a hunger_factor
        hunger += (hunger_factor/foodDistances[i]) * (foodGamma**i)

    # Beserk mode
    scaredGhosts = []
    for ghost in ghostStates:
        if ghost.scaredTimer > 0:
            md = manhattanDistance(ghost.getPosition(), pos)
            scaredGhosts.append(md)

    # Senzu bean
    capsuleDistances = []
    for capsule in capsules:
      md = manhattanDistance(capsule, pos)
      capsuleDistances.append(md)

    capsuleDistances = sorted(capsuleDistances)
    for i in range(len(capsuleDistances)):
        hunger += (hunger_factor*4/capsuleDistances[i]) * (foodGamma**i)

    scaredGhosts = sorted(scaredGhosts)
    scaredGhosts = [ghost for ghost in scaredGhosts if ghost < 5]
    for i in range(len(scaredGhosts)):
        hunger += (hunger_factor*2/scaredGhosts[i]) * (foodGamma**i)

    score =  hunger - fear + random.uniform(0, .5) - (n+7)**2 + currentGameState.getScore() - (len(capsules)+30)**2
    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
