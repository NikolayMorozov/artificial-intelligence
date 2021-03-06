"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""

import operator

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def heuristic1(game, player):
    """Calculate the heuristic value as a difference of number of legal moves available for player and its opponent.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    # getting opponent
    opponent = game.get_opponent(player)
    # obtaining locations
    playerMoves  = game.get_legal_moves(player)
    opponentMoves = game.get_legal_moves(opponent)
    

    # returning heuristic
    return float(len(playerMoves) - len(opponentMoves))
    
    
    
def heuristic2(game, player):
    """Calculate the heuristic value as a difference of number of legal moves available for player and its opponent and
     adding incentive to take central location.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    #Calculating center position of the game board 
    mid_w , mid_h = game.height // 2 + 1 , game.width // 2 + 1
    center_location  = (mid_w , mid_h)
    
    # getting players location
    player_location  = game.get_player_location(player)
    
    # checking if player is the center location
    if center_location == player_location:
        # returning heuristic1 with incentive 
        return heuristic1(game, player)+100
    else:
        # returning heuristic1 
        return heuristic1(game, player)

def proximity(location1, location2):
    '''
    Function return extra score as function of proximity between two positions.
    
    Parameters
    ----------
    location1, location2: tuple
        two tuples of integers (i,j) correspond to player location and center positon of the board

    Returns
    ----------
    float
        The heuristic value of 100 for center of the board position and zero otherwise   
    '''
    return abs(location1[0]-location2[0])+abs(location1[1]-location2[1])

def heuristic3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    opponent = game.get_opponent(player)
    player_location  = game.get_player_location(player)
    opponent_location = game.get_player_location(opponent) 
    playerMoves  = game.get_legal_moves(player)
    opponentMoves = game.get_legal_moves(opponent)

    
    blank_spaces = game.get_blank_spaces()
    board_size = game.width * game.height
    
    localArea = (game.width + game.height)/4
    
    if board_size - len(blank_spaces) > float(0.3 * board_size):

        playerMoves = [move for move in playerMoves if proximity(player_location, move)<=localArea]
        opponentMoves = [move for move in opponentMoves if proximity(opponent_location, move)<=localArea]
        
    return float(len(playerMoves) - len(opponentMoves))

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
             
    return heuristic3(game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        
        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        try:        
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            # pass
            
            # initializing variables that control iterative deepening, search algorithm and bestMove
            isID = self.iterative
            method = self.method
            bestValue = None
            
            
            if isID:   # block corresponding to iterative deepening
                # dict() to stor alternative solution as final selection method could contribute in to overall efficiency
                iterativeDeepening = dict()
                # starting with smalles depth
                depth = 1
                # infinte loop until algorithm doesn't run out 
                while True:
                    #calling appropriate search method
                    if  method == 'minimax':
                        score, move = self.minimax(game, depth, True)
                    elif method == 'alphabeta':                    
                        score, move = self.alphabeta(game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True)
                    
                    # adding newly received result into dictionary
#                    iterativeDeepening[move] = score
                    if move in iterativeDeepening:
                        if score > iterativeDeepening[move]:
                            iterativeDeepening[move] = score
                    else:
                        iterativeDeepening[move] = score
                    # getting move simply based best heuristic value. Perhaps,  not the cleverest way... 
                    bestValue = max(iterativeDeepening.items(), key=operator.itemgetter(1))[0]
                    
                    # returning current best option before time is out

                    if time_left() < 15:
                        return bestValue
                    
                    # incrementally increase search depth
                    depth +=1
                print('shit')
                return bestValue
            else: #fixed depth search branch
                depth = self.search_depth
                move = (-1, -1)
                # either minmax or alphabeta search method
                if method == 'minimax':
                    _, move = self.minimax(game, depth, True)
                elif method == 'alphabeta':                    
                    _, move = self.alphabeta(game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True)
                return move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return bestValue
            pass

        # Return the best move from the last completed search iteration
        print('hmmm', method, isID )
        return bestValue
        raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        

        # returning score value on depth zero  
        if depth == 0:             
            return self.score(game, self), game.get_player_location(self)            
        #initializing best move
        bestMove = (-1,-1)
        if maximizing_player:   #  maximizing player
            #initializing best value as negative infinity
            bestValue = float('-inf')
            #obtaining available legal moves
            legal_moves = game.get_legal_moves()
            #recursively evaluating moves
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                v, _ = self.minimax(nextGame, depth - 1, False) 
                # evaluating newly acquired score               
                if (v > bestValue ):
                    bestMove = move
                    bestValue = v
            # return score and move
            return float(bestValue), bestMove
        else:                   #  minimizing player
            #initializing best value as negative infinity
            bestValue = float('inf')
            #obtaining available legal moves
            legal_moves = game.get_legal_moves()
            #recursively evaluating moves
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                v, _ = self.minimax(nextGame, depth - 1, True)
                # evaluating newly acquired score 
                if (v < bestValue ):
                    bestMove = move
                    bestValue = v
            # return score and move
            return float(bestValue), bestMove

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # returning score value on depth zero 
        if depth == 0:
            s,m = self.score(game, self), game.get_player_location(self)             
            return s, m
        
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves)==0:
            bestMove = (-1, -1)
            s,m = self.score(game, self), game.get_player_location(self)
            return s, bestMove
        
        #initializing best move
        bestMove = (-1,-1)
        
        if maximizing_player:   # maximizer turn
            #obtaining available legal moves
            legal_moves = game.get_legal_moves()
            #recursively evaluating moves
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                v, m = self.alphabeta(nextGame, depth - 1, alpha=alpha, beta=beta, maximizing_player=False)
                if v > alpha:
                    alpha = v
                    bestMove = move
                #checking criteria for pruning
                if alpha >= beta:
                    return beta, bestMove
            return alpha, bestMove
        else:                   #minimizers turn
            #obtaining available legal moves
            legal_moves = game.get_legal_moves()
            #recursively evaluating moves
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                v, m = self.alphabeta(nextGame, depth - 1, alpha=alpha, beta=beta, maximizing_player=True)
                if v < beta:
                    beta = v
                    bestMove = move
                #checking criteria for pruning
                if beta <= alpha:
                    return beta, bestMove
            return beta, bestMove
