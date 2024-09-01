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
    label2.pack_forget()
    label3.pack_forget()
    b_bar.pack_forget()
    b_line.pack_forget()

def file_dialog():
    global data_path
    d_file_path=filedialog.askopenfilename()
    data_path.set(d_file_path)

def submit_bar():
    try:
        if len(titlevalue.get()) < 1:
            raise ValueError
        global lis
        lis = []
        lis_data = []
        lis_headings = []
        try:
            f = open(data_path.get(), "r")
        except:
            raise TypeError
        for a in f:
            if type(eval(a)) == type(lis_data):
                lis.append(eval(a))
                lis_data.append(eval(a))
            else:
                raise TypeError
        if len(lis_data) <1:
            raise TypeError
        for a in lis_data:
            if type(a[0]) == type(''):
                lis_headings.append(lis_data[lis_data.index(a)].pop(0))
            else:
                raise TypeError
        for a in lis_data:
            for b in a:
                if type(b) != type(1) and type(b) != type(1.1) and b != None:
                    raise TypeError
        screen4()
        directory_b()

    except TypeError:
        root_error = tk.Tk()
        label_error = tk.Label(root_error, text = "The file you gave is not \nas per the requirements. It must \nbe like the file 'SAMPLE.txt' \npresent in the same directory")
        label_error.place(anchor = "center", relx=0.5, rely=0.5)
        root_error.mainloop()

    except ValueError:
        root_error = tk.Tk()
        label_error = tk.Label(root_error, text = "Title field must not be empty")
        label_error.place(anchor = "center", relx=0.5, rely=0.5)
        root_error.mainloop()

def submit_line():
    try:
        if len(titlevalue.get()) < 1:
            raise NameError
        global lis
        lis = []
        lis_data = []
        lis_headings = []
        if x_min.get()>=x_max.get():
            raise ValueError
        try:
            f = open(data_path.get(), "r")
        except:
            raise TypeError
        for a in f:
            if type(eval(a)) == type(lis_data):
                lis.append(eval(a))
                lis_data.append(eval(a))
            else:
                raise TypeError
        if len(lis_data) <1:
            raise TypeError
        for a in lis_data:
            if type(a[0]) == type(''):
                lis_headings.append(lis_data[lis_data.index(a)].pop(0))
            else:
                raise TypeError
        for a in lis_data:
            for b in a:
                if type(b) != type(1) and type(b) != type(1.1) and b != None:
                    raise TypeError
        screen4_()
        directory_l()
    except TypeError:
        root_error = tk.Tk()
        label_error = tk.Label(root_error, text = "The file you gave is not \nas per the requirements. It must \nbe like the file 'SAMPLE.txt' \npresent in the same directory")
        label_error.place(anchor = "center", relx=0.5, rely=0.5)
        root_error.mainloop()

    except NameError:
        root_error = tk.Tk()
        label_error = tk.Label(root_error, text = "Title field must not be empty")
        label_error.place(anchor = "center", relx=0.5, rely=0.5)
        root_error.mainloop()

    except ValueError:
        root_error = tk.Tk()
        label_error = tk.Label(root_error, text = "Range min can't be more than \nor equal to max")
        label_error.place(anchor = "center", relx=0.5, rely=0.5)
        root_error.mainloop()

def screen4():
    label4.place_forget()
    label_title.grid_forget()
    label_data.grid_forget()
    titleentry.grid_forget()
    b_submit.grid_forget()
    b_data.grid_forget()
    label_data_value.grid_forget()

def directory_b():
    global label5, label6, label7, folder_path, b_browse, b_bargraph, filename
    label5 = tk.Label(text = "Step 3 out of 3", font = "Helvetica 16", bg = "black", fg = "white", padx = 30)
    label5.place(anchor = "center", relx =.5, rely = .04)
    filename = ""
    folder_path = tk.StringVar()
    tk.Label(text = "").grid(row = 0)
    tk.Label(text = "").grid(row = 1)
    label6 = tk.Label(text = "Choose the directory where you wish to store it!", font = ("comicsansms, 12"))
    label6.grid(row = 2)
    label7 = tk.Label(master=root,textvariable=folder_path)
    label7.grid(row = 3, column = 1)
    b_browse = tk.Button(text="Browse", command=browse)
    b_browse.grid(row = 3, column = 3)
    b_bargraph = tk.Button(text = "Plot the graph", command = bargraph)
    b_bargraph.place(anchor = "center", relx=.5, rely=.5)

def screen4_():
    screen4()
    label_x_min.grid_forget()
    entry_x_min.grid_forget()
    entry_x_max.grid_forget()
    label_x_max.grid_forget()

def directory_l():
    global label5, label6, label7, folder_path, b_browse, b_bargraph
    label5 = tk.Label(text = "Step 3 out of 3", font = "Helvetica 16", bg = "black", fg = "white", padx = 30)
    label5.place(anchor = "center", relx =.5, rely = .04)
    folder_path = tk.StringVar()
    tk.Label(text = "").grid(row = 0)
    tk.Label(text = "").grid(row = 1)
    label6 = tk.Label(text = "Choose the directory where you wish to store it!", font = ("comicsansms, 12"))
    label6.grid(row = 2)
    label7 = tk.Label(master=root,textvariable=folder_path)
    label7.grid(row = 3, column = 1)
    b_browse = tk.Button(text="Browse", command=browse)
    b_browse.grid(row = 3, column = 3)
    b_bargraph = tk.Button(text = "Plot the graph", command = linegraph)
    b_bargraph.place(anchor = "center", relx=.5, rely=.5)

def browse():
    global folder_path, filename
    filename = filedialog.askdirectory()
    folder_path.set(filename)

def bargraph():
    dircheck()
    bar_chart = pygal.Bar()
    bar_chart.title = titlevalue.get()
    maxcount = 0
    for i in range(len(lis)):
        bar_chart.add(lis[i][0], lis[i][1:])
    bar_chart.render_to_file(f"{filename}/graph.svg")
    webbrowser.open(f"{filename}/graph.svg")
    label5.place_forget()
    label6.grid_forget()
    label7.grid_forget()
    b_browse.grid_forget()
    b_bargraph.place_forget()
    label8 = tk.Label(text = "The graph has been plotted!", font = "Jokerman 25 bold", bg = "yellow", fg = "red", borderwidth = 6, relief = "ridge", padx = 20, pady = 10)
    label8.place(anchor = "center", relx = 0.5, rely = 0.5)

def linegraph():
    dircheck()
    line_chart = pygal.Line()
    line_chart.title = titlevalue.get()
    maxcount = 0
    for a in lis:
        if len(a) > maxcount:
            maxcount=len(a)
    for a in lis:
        while len(a) != maxcount:
            lis[a].insert(0, None)
    line_chart.x_labels = map(str, range(x_min.get(), x_max.get()))
    for a in range(len(lis)):
        line_chart.add(lis[a][0], lis[a][1:])
    line_chart.render_to_file(f"{filename}/graph.svg")
    webbrowser.open(f"{filename}/graph.svg")
    label5.place_forget()
    label6.grid_forget()
    label7.grid_forget()
    b_browse.grid_forget()
    b_bargraph.place_forget()
    label8 = tk.Label(text = "The graph has been plotted!", font = "Jokerman 25 bold", bg = "yellow", fg = "red", borderwidth = 6, relief = "ridge", padx = 20, pady = 10)
    label8.place(anchor = "center", relx = 0.5, rely = 0.5)

def dircheck():
    global filename
    if filename == "":
        root_error = tk.Tk()
        label_error = tk.Label(root_error, text = "Filepath must not be empty")
        label_error.pack(fill = "both")
        root_error.mainloop()

img1 = Image.open("Images/start.png")
n_img1 = img1.resize((500, 200))
nimg1 = ImageTk.PhotoImage(image = n_img1)
label1 = tk.Label(text = "Welcome to graph plotter!", font = "Jokerman 25 bold", bg = "yellow", fg = "red", borderwidth = 6, relief = "ridge", padx = 20, pady = 10)
label1.pack(fill = "both", side = "top", padx = 20, pady = 10)
b_start = tk.Button(image = nimg1,borderwidth=3, command = screen2)
b_start.pack()

#Main-loop
root.mainloop()
