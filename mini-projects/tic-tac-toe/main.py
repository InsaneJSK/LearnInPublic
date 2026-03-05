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
    marker = 'AXAXAXAXAZAZA'

    while marker not in ('x' or 'o'):
        marker = input('Player 1: Do you want to be x or o? ').upper()
        if marker == 'X':
            return ('X', 'O')
        elif marker == 'O':
            return ('O', 'X')

def place_marker(board, marker, pos):
    """
    This one will add a marker into your desired position.
    """
    board[pos] = marker

def win_check(board, mark):
    """
    This one will check wether you won or not
    """
    return ((board[1] == board[2] == board[3] == mark) or
            (board[4] == board[5] == board[6] == mark) or
            (board[7] == board[8] == board[9] == mark) or
            (board[1] == board[4] == board[7] == mark) or
            (board[2] == board[5] == board[8] == mark) or
            (board[3] == board[6] == board[9] == mark) or
            (board[7] == board[5] == board[3] == mark) or
            (board[9] == board[5] == board[1] == mark))

def choose_first():
    """
    This one will decide that who will go first
    """
    number = random.randint(0, 1)

    if number == 0:
        return 'Player 2'
    else:
        return 'Player 1'

def space_check(board, pos):
    """
    This one will check that is there any space at the position you chose.
    """
    return board[pos] == ' '

def full_board_check(board):
    """
    This one will check for a tie.
    """
    return (board[1] != ' ' and
            board[2] != ' ' and
            board[3] != ' ' and
            board[4] != ' ' and
            board[5] != ' ' and
            board[6] != ' ' and
            board[7] != ' ' and
            board[8] != ' ' and
            board[9] != ' ')

def player_choice(board):
    """
    This one will ask you that where do you want to place your marker
    """
    pos = 0
    while pos not in [1, 2, 3, 4, 5, 6, 7, 8, 9] or not space_check(board, pos):
        pos = int(input('Choose a position 1-9 '))

    return pos

def replay():
    """
    This one will ask you that you want to play again or not.
    """
    wanna = 'jyyfhv'
    while wanna not in ['Y', 'N']:
        wanna = input("Wanna play again(ans in Y or N) :")

    return wanna == 'N'

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
