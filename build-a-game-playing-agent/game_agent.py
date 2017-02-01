"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

# for test purpose only!!!!
import isolation

import agent_test
import operator
import time




class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


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
    oponent = game.get_opponent(player)
    blank_spaces = game.get_blank_spaces() 
    playerMoves  = game.get_legal_moves(player)
    oponentMoves = game.get_legal_moves(oponent)
         
    return float(len(blank_spaces)+len(playerMoves)-len(oponentMoves))

class searchNode:
    def __init__(self, game, parent, move, lvl):
        
        self.state = game
        self.parentState = parent
        self.movedTo = move
        self.searchLevel = lvl




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
        
        startTime = time.time()
        iterativeDeepening = dict()
       

        
        
        
        self.time_left = time_left
        
        startTime = time.time()
        max_search_depth = self.search_depth 
        
        
#        print('iterative', self.iterative)
#        print('method', self.method)

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
#            print('SI', self.iterative)
            bestValue = None
#            if self.iterative == True:
            if True:
                flag = True
                depth = 1
                while flag:
#                    print(depth)
                    s, m = self.minimax(game, depth, True)
                    iterativeDeepening[m] = s

                    bestValue = max(iterativeDeepening.items(), key=operator.itemgetter(1))[0]
                    
                    if (time.time()-startTime) > 0.95:
#                        print('it is showtime')
#                        flag = False 
                        return bestValue
                    depth +=1
            else:
                pass

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
#        return bestMove
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
        
        node_terminal = False

        if (depth == 0)  or node_terminal:
            s,m = self.score(game, self), game.get_player_location(self)             
            return s, m             

        if maximizing_player: #  maximizing player
            bestValue = float('-inf')
            bestMove = None
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                v, m = self.minimax(nextGame, depth - 1, False)                
                if (v > bestValue ):
                    bestMove = move
                    bestValue = v
            return float(bestValue), bestMove

        else:  #  minimizing player
            bestValue = float('inf')
            bestMove = None
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                v, m = self.minimax(nextGame, depth - 1, True)
                if (v < bestValue ):
                    bestMove = move
                    bestValue = v
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

        print('========================')
        print('d',depth)
   # check if at search bound


        
#        isTerminal = False
        if depth == 0:
            s,m = self.score(game, self), game.get_player_location(self)             
            return s, m
        
        legal_moves = game.get_legal_moves()
        
        if len(legal_moves)==0:
            bestMove = (-1, -1)
            s,m = self.score(game, self), game.get_player_location(self)
            return s, bestMove
            

#        children = successors(node)

        bestMove = (-1,-1)


        if maximizing_player:   # maximizer turn
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                v, m = self.alphabeta(nextGame, depth - 1, alpha=alpha, beta=beta, maximizing_player=False)
                if v > alpha:
                    alpha = v
                    bestMove = move
                if alpha > beta:
                    return beta, move
            return alpha, move
        else:                   #minimizers turn
            legal_moves = game.get_legal_moves()
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                v, m = self.alphabeta(nextGame, depth - 1, alpha=alpha, beta=beta, maximizing_player=True)
                if v < beta:
                    beta = v
                if beta < alpha:
                    return beta, move
            return beta, move

        # TODO: finish this function!
        raise NotImplementedError

if __name__ == '__main__':
        player1 = "Player1"
        player2 = "Player2"
        p1_location = (5, 5)
        p2_location = (1, 1)  # top left corner
        game = isolation.Board(player1, player2)
        game.apply_move(p1_location)
        game.apply_move(p2_location)

        print('c_score', custom_score(game, player1))
        
        at = agent_test.Project1Test()

        
     #   at.test_heuristic()
     #   at.test_get_move_interface()
        
     #   at.test_minimax_interface()
     #   at.test_minimax()
        
#        at.test_get_move()
        
        at.test_alphabeta_interface()
        at.test_alphabeta()
