import random, os
from pynput import keyboard
from pynput.keyboard import Key

lis = [[" ", " ", " ", " ", " "], [" ", " ", " ", " ", " "], [" ", " ", " ", " ", " "], [" ", " ", " ", " ", " "], [" ", " ", " ", " ", " "]]

#Functions
def board():
    for i in lis:
        for j in i:
            print("|", end = "")
            print(str(j).center(5), end = "|")
        print()

def selector():
    varx = random.randint(0, 4)
    vary = random.randint(0, 4)
    if lis[vary][varx] == " ":
        lis[vary][varx] = num()
    else:
        selector()

def num():
    var = random.randint(1, 2)
    if var == 1:
        return 2
    else:
        return 4

def Check():
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
    pass

def Left():
    pass

def on_key_release(key):
    if key == Key.right:
        Right()
    elif key == Key.left:
        Left()
    elif key == Key.up:
        Up()
    elif key == Key.down:
        Down()

#main
selector()
board()
print()

with keyboard.Listener(on_release=on_key_release) as listener:
    listener.join()
