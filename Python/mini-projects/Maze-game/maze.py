import os
from pynput import keyboard
from pynput.keyboard import Key

mazelayout = [["X", "|", "END", "|", "X", "|", "X"],
        ["X", "|", " ", "|", "X", "|", "X"],
        ["X", "|", " ", "|", " ", "|", "X"],
        ["X", "|", "X", "|", " ", "|", " "]]

O_Val = [3, 6]
N_Val = O_Val
Boolean = False

def maze():
    clear()
    global Boolean
    x,y = N_Val
    if mazelayout[x][y] != "END":
        mazelayout[x][y] = "O"
        mazeupdate()
    else:
        print("Congrats you won!")
        quit()

def mazeupdate():
    for i in mazelayout:
        for j in i:
            print(f"{j:4}", end = "")
        print()
    print()

def moveleft():
    global N_Val
    x, y = N_Val
    if mazelayout[x][y] == "O":
        mazelayout[x][y] = " "
        if y >= 2 and mazelayout[x][y-2] != "X":
            y -= 2
    N_Val = [x, y]
    maze()

def moveright():
    global N_Val
    x, y = N_Val
    if mazelayout[x][y] == "O":
        mazelayout[x][y] = " "
        if y <= 4 and mazelayout[x][y+2] != "X":
            y += 2
    N_Val = [x, y]
    maze()

def moveup():
    global N_Val
    x, y = N_Val
    if mazelayout[x][y] == "O":
        mazelayout[x][y] = " "
        if x >= 1 and mazelayout[x-1][y] != "X":
            x -= 1
    N_Val = [x, y]
    maze()

def movedown():
    global N_Val
    x, y = N_Val
    if mazelayout[x][y] == "O":
        mazelayout[x][y] = " "
        if x <= 2 and mazelayout[x+1][y] != "X":
            x += 1
    N_Val = [x, y]
    maze()

def clear():
    os.system("cls")
   
def on_key_release(key):
    if key == Key.right:
        moveright()
    elif key == Key.left:
        moveleft()
    elif key == Key.up:
        moveup()
    elif key == Key.down:
        movedown()
    elif key == Key.esc:
        quit()


maze()

with keyboard.Listener(on_release=on_key_release) as listener:
    listener.join()
