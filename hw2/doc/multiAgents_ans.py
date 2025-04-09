from cmath import inf
from pkg_resources import add_activation_listener
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """


    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        pacmanIndex = 0
        def minimax(depth, agentIndex, gameState):
            # if the game state is win, lose or the game reach a deeper layer
            # then we evaluate the score.
            if(gameState.isWin() or gameState.isLose() or depth > self.depth):
                return self.evaluationFunction(gameState)
            
            scores = []
            # we get all the legal actions for the agentIndex,
            # and remove the stop action
            actions = gameState.getLegalActions(agentIndex)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            # find all the nextStates based on the action
            # using recursion call        
            for action in actions:
                nextState = gameState.getNextState(agentIndex, action)
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    scores.append(minimax(depth+1, 0, nextState))
                else:
                    scores.append(minimax(depth, nextAgent, nextState))
            # if the agentIndex = 0 -> pacman, which is a max player
            # if the agentIndex > 0 -> ghost, which is a min player
            if agentIndex == 0:
                # if we reach the first layer, then we determine the action
                # based on the score calculated in the deeper layer
                if depth == 1:
                    max_score = max(scores)
                    idx = scores.index(max_score)
                    return actions[idx]
                else:
                    return max(scores)
            else:
                return min(scores)
        return minimax(1, pacmanIndex, gameState)
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        pacmanIndex = 0
        # set default alpha as negative infinite
        # and default beta as positive infinite
        alpha = -inf
        beta = inf
        def alphaBeta(depth, agentIndex, gameState, alpha, beta):
            # if the game state is win, lose or the game reach a deeper layer
            # then we evaluate the score.
            if(gameState.isWin() or gameState.isLose() or depth > self.depth):
                return self.evaluationFunction(gameState)
            
            scores = []
            # we get all the legal actions for the agentIndex,
            # and remove the stop action
            actions = gameState.getLegalActions(agentIndex)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)

            # find all the nextStates based on the action
            # using recursion call
            # and constantly update alpha, beta using value variable 
            # to avoid unnecessary computing
            # also add value variable into the score                    
            for action in actions:
                nextState = gameState.getNextState(agentIndex, action)
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    value = -inf
                    value = max(value, alphaBeta(depth+1, 0, nextState, alpha, beta))
                else:
                    value = inf
                    value = min(value, alphaBeta(depth, nextAgent, nextState, alpha, beta))
                
                if agentIndex == 0:
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                else:
                    if value < alpha:
                        return value
                    beta = min(beta, value)

                scores.append(value)
            # if the agentIndex = 0 -> pacman, which is a max player
            # if the agentIndex > 0 -> ghost, which is a min player            
            if agentIndex == 0:
                # if we reach the first layer, then we determine the action
                # based on the score calculated in the deeper layer
                if depth == 1:
                    max_score = max(scores)
                    idx = scores.index(max_score)
                    return actions[idx]
                else:
                    return max(scores)
            else:
                return min(scores)
        return alphaBeta(1, pacmanIndex, gameState, alpha, beta)
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        pacmanIndex = 0
        def expectimax(depth, agentIndex, gameState):
            # if the game state is win, lose or the game reach a deeper layer
            # then we evaluate the score.
            if(gameState.isWin() or gameState.isLose() or depth > self.depth):
                return self.evaluationFunction(gameState)
            scores = []
            # we get all the legal actions for the agentIndex,
            # and remove the stop action            
            actions = gameState.getLegalActions(agentIndex)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            # find all the nextStates based on the action
            # using recursion call                     
            for action in actions:
                nextState = gameState.getNextState(agentIndex, action)
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    scores.append(expectimax(depth+1, 0, nextState))
                else:
                    scores.append(expectimax(depth, nextAgent, nextState))
            # if the agentIndex = 0 -> pacman, which is a max player
            # if the agentIndex > 0 -> ghosts, we calculate the expectation
            # of the scores             
            if agentIndex == 0:
                # if we reach the first layer, then we determine the action
                # based on the score calculated in the deeper layer
                if depth == 1:
                    max_score = max(scores)
                    idx = scores.index(max_score)
                    return actions[idx]
                else:
                    return max(scores)
            else:
                total = sum(scores)
                expectation = float(total / len(scores))
                return expectation
        return expectimax(1, pacmanIndex, gameState)
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    # get the pacman's position
    pos = currentGameState.getPacmanPosition()
    # get all the food's position
    foodList = currentGameState.getFood().asList()
    # get all the ghost's poistion and there distance to pacman
    ghosts = currentGameState.getGhostPositions()
    ghostsDis = [manhattanDistance(ghosts[i], pos) for i in range(len(ghosts))]
    # get all the capsules's position
    capsules = currentGameState.getCapsules()
    
    # distance to eat the nearest food.
    foodDis = 100
    for food in foodList:
        foodDis = min(manhattanDistance(pos, food), foodDis)
    # if the distance to the food is closer, ther the score is higher
    foodScore = 100 - foodDis
    
    # distance to all the ghosts and check whether the ghost is scared
    ghostsScore = []
    for i in range(len(ghostsDis)):
        ghostTimer = currentGameState.getGhostStates()[i].scaredTimer
        # if the ghost is scared, and the distance is closer, then the score is higher
        # otherwise, when the distance is close, the score is lower
        if ghostTimer > 0:
            ghostsScore.append(100 - ghostsDis[i]) 
        else:
            ghostsScore.append(-(100 - ghostsDis[i]))
    ghostScore = max(ghostsScore)
    #distance to all the capsules
    capDis = 100
    for capsule in capsules:
        capDis = min(manhattanDistance(pos, capsule), capDis)
    # if the distance to the capsule is closer, then the score is higher 
    capScore =  100 - capDis
    # The total score is calculated by the the original score plus foodScore, ghostScore, capScore
    score = currentGameState.getScore() + foodScore + ghostScore + capScore
    return score
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
