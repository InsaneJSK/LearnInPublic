"""
we're gonna play TIK_TAK_TOE today
"""

import random
import os

#Functions
CLEAR = lambda: os.system('cls')
"""
THIS WILL HELP CLEAR THE SCREEN
"""

def display_board(board):
    """
    this will display a board for the game.
    """
    CLEAR()
    print(board[1] + ' | ' + board[2] + ' | ' + board[3])
    print(board[4] + ' | ' + board[5] + ' | ' + board[6])
    print(board[7] + ' | ' + board[8] + ' | ' + board[9])

def player_input():
    """
    This will take the marker the player will use in the whole game
    """
    pass

def choose_first():
    """
    This one will decide that who will go first
    """
    pass

def player_choice():
    """
    This one will ask you that where do you want to place your marker
    """
    pass

def full_board_check():
    """
    This one will check for a tie.
    """
    pass

def place_marker():
    """
    This one will add a marker into your desired position.
    """
    pass

def win_check():
    """
    This one will check wether you won or not
    """
    pass

def replay():
    """
    This one will ask you that you want to play again or not.
    """
    pass

#Main
print('Welcome to Tic Tac Toe!')

LETS_PLAY = True

while LETS_PLAY:
    THE_BOARD = [' '] * 10
    PLAYER_1_MARKER, PLAYER_2_MARKER = player_input()
    TURN = choose_first()
    print(f'{TURN} will go first')

    PLAY_GAME = ''
    while PLAY_GAME not in ['y', 'n']:
        PLAY_GAME = input('Ready to play? y or n')

    GAME_ON = PLAY_GAME == 'y'

    while GAME_ON:
        if TURN == 'Player1':
            display_board(THE_BOARD)
            POSITION = player_choice(THE_BOARD)
            place_marker(THE_BOARD, PLAYER_1_MARKER, POSITION)

            if win_check(THE_BOARD, PLAYER_1_MARKER):
                display_board(THE_BOARD)
                print('player1 has won')
                GAME_ON = False

            else:
                if full_board_check(THE_BOARD):
                    display_board(THE_BOARD)
                    print('tie game')
                    GAME_ON = False
                else:
                    TURN = 'Player2'
        else:
            display_board(THE_BOARD)
            POSITION = player_choice(THE_BOARD)
            place_marker(THE_BOARD, PLAYER_2_MARKER, POSITION)

            if win_check(THE_BOARD, PLAYER_2_MARKER):
                display_board(THE_BOARD)
                print('player2 has won')
                GAME_ON = False

            else:
                if full_board_check(THE_BOARD):
                    display_board(THE_BOARD)
                    print('tie game')
                    GAME_ON = False
                else:
                    TURN = 'Player1'
    if replay():
        LETS_PLAY = False
