"""Class 12 project on Library management software by Jaspreet Singh"""
import mysql.connector as sqltor
from tabulate import tabulate

#SQL Database
mycon = sqltor.connect(host = "localhost", user = "root", password = "jaspreet", auth_plugin='mysql_native_password')
if mycon.is_connected():
    print("Connection secured")
cursor = mycon.cursor()
cursor.execute("create database if not exists library")
cursor.execute("use library")
creat_tb = "create table if not exists books (BCode char(3) primary key, BName varchar(50) not null, AuthorName varchar(30) NOT Null, Borrower varchar(30), IssueDate date, Returndate date)"
cursor.execute(creat_tb)
vals = """insert into books(BCode, Bname, AuthorName, Borrower, IssueDate, ReturnDate) values ('001', 'The power of your sub-concious mind', 'Joseph Murphy', 'Abc', '2023-01-01', NULL),
        ('002', 'Atomic Habits', 'James Clear', NULL, '2022-12-24', '2022-12-30'),
        ('003', 'Keep Going', 'Austin Kleon', 'Cde', '2023-01-02', NULL),
        ('004', 'Show your work', 'Austin Kleon', NULL, '2022-10-01', '2022-10-09'),
        ('005', 'Steal like an artist', 'Austin Kleon', 'Efg', '2023-01-01', NULL),
        ('006', 'The law of success', 'Napolean Hill', NULL, '2022-12-15', '2022-12-19'),
        ('007', 'Think and grow rich', 'Napolean Hill', 'Ghi', '2023-01-01', NULL),
        ('008', 'Accessory to War', 'Neil deGrasse Tyson', 'Ijk', '2023-12-01', NULL),
        ('009', 'Astrophysics for people in a hurry', 'Neil deGrasse Tyson', NULL, '2023-01-04', '2023-01-12'),
        ('010', 'Chaos: Making a new science', 'James Gleick', 'Klm', '2023-01-04', NULL)"""
cursor.execute(vals)

#Functions
def menu():
    print("Choose the action by pressing the number next to the command")
    print("1 Show the books")
    print("2 Add a new book")
    print("3 Delete an existing book")
    print("4 Update an existing book")
    print("5 Find a specific book")
    print("6 Exit")
    loop = True
    while loop == True:
        var = input("Enter the number: ")
        if var == "1":
            loop = False
            show()
        elif var == "2":
            loop = False
            add()
        elif var == "3":
            loop = False
            delete()
        elif var == "4":
            loop = False
            update()
        elif var == "5":
            loop = False
            find()
        elif var == "6":
            print("Exitting the application")
            quit()
        else:
            print("That's not an appropriate choice, try again")

def show():
    print("What do you wish to see")
    print("1 Show all")
    print("2 Show borrowed books")
    print("3 Show books not borrowed")
    print("4 Back")
    print("5 Exit")
    loop1 = True
    while loop1:
        var1 = input("Enter the number: ")
        if var1 == "1":
            loop1 = False
            showall()
        elif var1 == "2":
            loop1 = False
            showborrowed()
        elif var1 == "3":
            loop1 = False
            shownotborrowed()
        elif var1 == "4":
            loop1 = False
            menu()
        elif var1 == "5":
            print("Exitting the application")
            quit()
        else:
            print("That's not an appropriate choice, try again")

def showall():
    cursor.execute("select * from books")
    db = cursor.fetchall()
    db = list(db)
    table(db)
    input("-----Press enter to continue-----")
    menu()

def showborrowed():
    cursor.execute("select * from books where ReturnDate is NULL;")
    db = cursor.fetchall()
    db = list(db)
    table(db)
    input("-----Press enter to continue-----")
    menu()

def add():
    pass

def delete():
    pass

def update():
    pass

def find():
    pass

def shownotborrowed():
    pass

def table():
    pass
