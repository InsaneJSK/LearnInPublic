import random, os
from pynput import keyboard
from pynput.keyboard import Key

#This is our game board
lis = [[" ", " ", " ", " ", " "], [" ", " ", " ", " ", " "], [" ", " ", " ", " ", " "], [" ", " ", " ", " ", " "], [" ", " ", " ", " ", " "]]

#Functions
def board():
    """
    Printing the game board
    """
    for i in lis:
        for j in i:
            print("|", end = "")
            print(str(j).center(5), end = "|")
        print()

def selector():
    """
    This code will select random X-Y coordinates in order to choose the place for putting the randomly generated number
    """
    varx = random.randint(0, 4)
    vary = random.randint(0, 4)
    if lis[vary][varx] == " ":
        lis[vary][varx] = num()
    else:
        selector()

def num():
    """
    This code will use random.randint to choose whether to add 2 or 4 on the selected space
    """
    var = random.randint(1, 2)
    if var == 1:
        return 2
    else:
        return 4

def Check():
    """
    This code will check if you won or lost the game
    """
    boole = True
    for i in lis:
        for j in i:
            if j == 2048:
                print("Congrats!! You won!")
                quit()
            if j == " ":
                boole = False
    if boole:
        print("Game over!")
        quit()

def Down():
    """
    This code will be moving all the numbers and merging downwards
    """
    for i in range(4):
        for j in range(5):
            if lis[i][j] != " " and lis[i+1][j] == " ":
                lis[i][j], lis[i+1][j] = lis[i+1][j], lis[i][j]

    for i in range(4, 0, -1):
        for j in range(4, -1, -1):
            if lis[i][j] == lis[i-1][j] and lis[i][j] != " ":
                lis[i][j], lis[i-1][j] = 2*lis[i][j], " "

    for i in range(4):
        for j in range(5):
            if lis[i][j] != " " and lis[i+1][j] == " ":
                lis[i][j], lis[i+1][j] = lis[i+1][j], lis[i][j]

    selector()
    os.system("cls")
    board()
    Check()

def Up():
    """
    This code will be moving all the numbers and merging upwards
    """
    for i in range(4, 0, -1):
        for j in range(4, -1, -1):
            if lis[i][j] != " " and lis[i-1][j] == " ":
                lis[i][j], lis[i-1][j] = lis[i-1][j], lis[i][j]
    for i in range(4):
        for j in range(5):
            if lis[i][j] == lis[i+1][j] and lis[i][j] != " ":
                lis[i][j], lis[i+1][j] = 2*lis[i][j], " "
    for i in range(4, 0, -1):
        for j in range(4, -1, -1):
            if lis[i][j] != " " and lis[i-1][j] == " ":
                lis[i][j], lis[i-1][j] = lis[i-1][j], lis[i][j]
    selector()
    os.system("cls")
    board()
    Check()

def Right():
    """
    This code will be moving all the numbers and merging towards right
    """
    for i in range(5):
        for j in range(4):
            if lis[i][j] != " " and lis[i][j+1] == " ":
                lis[i][j], lis[i][j+1] = lis[i][j+1], lis[i][j]
    for i in range(4, -1, -1):
        for j in range(4, 0, -1):
            if lis[i][j] == lis[i][j-1] and lis[i][j] != " ":
                lis[i][j], lis[i][j-1] = 2*lis[i][j], " "
    for i in range(5):
        for j in range(4):
            if lis[i][j] != " " and lis[i][j+1] == " ":
                lis[i][j], lis[i][j+1] = lis[i][j+1], lis[i][j]
    selector()
    os.system("cls")
    board()
    Check()

def Left():
    """
    This code will be moving all the numbers and merging towards left
    """
    for i in range(4, -1, -1):
        for j in range(4, 0, -1):
            if lis[i][j] != " " and lis[i][j-1] == " ":
                lis[i][j], lis[i][j-1] = lis[i][j-1], lis[i][j]
    for i in range(5):
        for j in range(4):
            if lis[i][j] == lis[i][j+1] and lis[i][j] != " ":
                lis[i][j], lis[i][j+1] = 2*lis[i][j], " "
    for i in range(4, -1, -1):
        for j in range(4, 0, -1):
            if lis[i][j] != " " and lis[i][j-1] == " ":
                lis[i][j], lis[i][j-1] = lis[i][j-1], lis[i][j]
    selector()
    os.system("cls")
    board()
    Check()

def on_key_release(key):
    """
    This will use pynput to determine if the user pressed left right up or down
    """
    if key == Key.right:
        Right()
    elif key == Key.left:
        Left()
    elif key == Key.up:
        Up()
    elif key == Key.down:
        Down()
    elif key == Key.esc:
        try:
            quit()
        except KeyboardInterrupt:
            print("Exiting the game")

#main
print("""
      Welcome!
      This is the classic game 2048
      in python!!!
      Commands are simple, use the arrow keys
      If you wanna quit, press escape
      Enjoy
      """)
selector()
board()
print()

with keyboard.Listener(on_release=on_key_release) as listener:
    listener.join()
