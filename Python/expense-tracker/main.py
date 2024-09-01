"""Input ny questions or queries you have for the environment. This bpt will provide you the bst possible answers for the same.""" 

import tkinter as tk

#Files
f1 = open("ques.txt", "r")
f2 = open("ans.txt", "r")

#ROOT
root = tk.Tk()

#Title
root.title("LifeBot")
#icon = tk.PhotoImage(file = "Images/icon.png")
#root.iconphoto(False, icon)

#Geometry
root.geometry("850x800")
root.minsize(850, 600)
root.maxsize(850, 600)

#Lists
quesnum = None
queslis = []
for i in f1.readlines():
    try:
        queslis.append(i)
    except EOFError:
        break

anslis = []
for i in f2.readlines():
    try:
        anslis.append(i)
    except EOFError:
        break

#Functions
def known_questions(questions):
    for i in queslis:        
        y = i.split()
        num = y[0]
        z = ""
        for j in y[1::]:
            z += f"{j} "
        if questions == z[:-1:]:
            quesnum = int(num)
    return quesnum

def unknown_questions(question):
    kwlist = []
    quesword = str(question).split()
    for i in queslis:        
        y = i.split()
        ctr = 0
        for j in quesword:
            if j in y and j.lower() not in ["what", "how", "is", "a", "the", "of", "are", "does", "can"]:
                ctr += 1
        kwlist.append(ctr)
    var = max(kwlist)
    simques = []
    ctr2 = 0
    for k in kwlist:
        ctr2 += 1
        if k == var:
            simques.append(ctr2)
    return simques

def answers(var):
    ans = ""
    for i in anslis[var-1].split()[1::]:
        ans += f"{i} "
    return ans

def question(var):
    ques = ""
    for i in queslis[var-1].split()[1::]:
        ques += f"{i} "
    return ques

def quesmain():
    global quesvalue, labell
    label1.pack_forget()
    b_start.pack_forget()
    label2 = tk.Label(text = "Enter your questions:", font = "Gabriola 20")
    label2.pack(side = 'top', fill = 'both')
    quesvalue = tk.StringVar()
    quesentry = tk.Entry(root, textvariable = quesvalue, width = 50)
    quesentry.pack(side = 'top', fill = 'both')
    b_generate = tk.Button(text = "Generate response", borderwidth = 3, command = ansmain)
    b_generate.pack(side = 'top', fill = 'both')
    b_exit = tk.Button(text = "Exit application", borderwidth = 3, command = exit)
    b_exit.pack(side = 'left', fill = 'both')
    labell = tk.Label(text = "")
    labell.pack(side = 'top', fill = 'both')

def ansmain():
    global labell
    labell.destroy()
    try:
        num = known_questions(quesvalue.get())
        labell = tk.Label(text = answers(num), wraplength = 500)
        labell.pack(side = 'bottom')
    except UnboundLocalError:
        lis = unknown_questions(quesvalue.get())
        unans = "Couldn't find the exact answer, finding some similar questions...\n\n"
        for i in lis:
            unans += f"{question(i)}\n{answers(i)}\n\n"
        labell = tk.Label(text = unans, wraplength = 500)
        
        labell.pack(side = 'bottom')

def exit():
    root.quit()

label1 = tk.Label(text = "Welcome to the lifebot!\nAsk your questions and we\nwill answer them for you!!", font = "Gabriola 30 bold", bg = "pink", fg = "green", borderwidth = 6, relief = "ridge", padx = 20, pady = 10)
label1.pack(fill = "both", padx = 20, pady = 50)
b_start = tk.Button(text = "Start", font = "comicsan 15 bold", borderwidth=3, command = quesmain)
b_start.pack()  


#Main-loop
root.mainloop()
