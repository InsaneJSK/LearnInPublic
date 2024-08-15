import pygal, webbrowser
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

#ROOT
root = tk.Tk()

#Title
root.title("Graph Plotter")
icon = tk.PhotoImage(file = "Images/icon.png")
root.iconphoto(False, icon)

#Geometry
root.geometry("600x400")
root.minsize(600, 400)
root.maxsize(600, 400)

#Functions
def screen2():
    label1.pack_forget()
    b_start.pack_forget()

    #Labels
    global label2, label3, b_bar, b_line
    label2 = tk.Label(text = "Step 1 out of 3", font = "Helvetica 20", bg = "black", fg = "white", padx = 30)
    label2.pack()
    label3 = tk.Label(text = "Choose the type of graph you want: ", font = "comicsansms 15")
    label3.pack(pady = 30)
    b_bar = tk.Button(text = "Bar Graph", command = bar, font = "comicsansms 15")
    b_bar.pack(pady = 30)
    b_line = tk.Button(text = "Line Graph", command = line, font = "comicsansms 15")
    b_line.pack(pady = 30)

def bar():
    screen3()
    global label4, label_title, label_data, titlevalue, titleentry, b_data, b_submit, data_path, label_data_value
    titlevalue = tk.StringVar()
    data_path = tk.StringVar()
    label4 = tk.Label(text = "Step 2 out of 3", font = "Helvetica 20", bg = "black", fg = "white", padx = 30)
    label4.place(anchor = "center", relx =.5, rely = .04)
    tk.Label(text = "").grid(row = 0)
    tk.Label(text = "").grid(row = 1)
    label_title = tk.Label(text = "Title", padx = 75, font = ("comicsansms, 15"))
    label_title.grid(row = 2, pady = 30)
    titleentry = tk.Entry(root, textvariable = titlevalue, font = ("comicsansms, 15"))
    titleentry.grid(row = 2, column = 1, pady = 30)
    label_data = tk.Label(text = "Data\n(As a list)", font = ("comicsansms, 15"))
    label_data.grid(row = 3, pady = 30)
    label_data_value = tk.Label(master=root, textvariable=data_path, font = ("comicsansms, 15"))
    label_data_value.grid(row = 3, column = 1, pady = 30)
    b_data = tk.Button(text="Choose File", command=file_dialog, font = ("comicsansms, 15"))
    b_data.grid(row = 3, column = 2, pady = 30)
    b_submit = tk.Button(text = "Submit", command = submit_bar, font = ("comicsansms, 15"))
    b_submit.grid(row = 4, column = 1, pady = 30)

def line():
    screen3()
    global label4, label_title, label_data, titlevalue, titleentry, b_data, b_submit, data_path, label_data_value, x_min,    x_max, label_x_min, entry_x_min, label_x_max, entry_x_max
    titlevalue = tk.StringVar()
    data_path = tk.StringVar()
    x_min = tk.IntVar()
    x_max = tk.IntVar()
    label4 = tk.Label(text = "Step 2 out of 3", font = "Helvetica 16", bg = "black", fg = "white", padx = 30)
    label4.place(anchor = "center", relx =.5, rely = .04)
    tk.Label(text = "").grid(row = 0)
    tk.Label(text = "").grid(row = 1)
    label_title = tk.Label(text = "Title", padx = 75, font = ("comicsansms, 15"))
    label_title.grid(row = 2)
    titleentry = tk.Entry(root, textvariable = titlevalue, font = ("comicsansms, 12"))
    titleentry.grid(row = 2, column = 1)
    label_x_min = tk.Label(text = "Horizontal range minimum value", font = ("comicsansms, 15"))
    label_x_min.grid(row = 3, column = 0)
    entry_x_min = tk.Entry(root, textvariable = x_min, font = ("comicsansms, 12"))
    entry_x_min.grid(row = 3, column = 1)
    label_x_max = tk.Label(text = "Horizontal range maximum value", font = ("comicsansms, 15"))
    label_x_max.grid(row = 4, column = 0)
    entry_x_max = tk.Entry(root, textvariable = x_max, font = ("comicsansms, 12"))
    entry_x_max.grid(row = 4, column = 1)
    label_data = tk.Label(text = "Data\n(As a list)", font = ("comicsansms, 15"))
    label_data.grid(row = 5)
    label_data_value = tk.Label(master=root,textvariable=data_path, font = ("comicsansms, 15"))
    label_data_value.grid(row = 5, column = 1)
    b_data = tk.Button(text="Choose File", command=file_dialog, font = ("comicsansms, 15"))
    b_data.grid(row = 5, column = 2)
    b_submit = tk.Button(text = "Submit", command = submit_line, font = ("comicsansms, 15"))
    b_submit.grid(row = 6, column = 1)

def screen3():
    pass

def file_dialog():
    pass

def submit_bar():
    pass

def submit_line():
    pass

img1 = Image.open("Images/start.png")
n_img1 = img1.resize((500, 200))
nimg1 = ImageTk.PhotoImage(image = n_img1)
label1 = tk.Label(text = "Welcome to graph plotter!", font = "Jokerman 25 bold", bg = "yellow", fg = "red", borderwidth = 6, relief = "ridge", padx = 20, pady = 10)
label1.pack(fill = "both", side = "top", padx = 20, pady = 10)
b_start = tk.Button(image = nimg1,borderwidth=3, command = screen2)
b_start.pack()

#Main-loop
root.mainloop()