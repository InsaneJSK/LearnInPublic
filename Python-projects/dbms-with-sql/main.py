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

def shownotborrowed():
    cursor.execute("select * from books where ReturnDate is not Null;")
    db = cursor.fetchall()
    db = list(db)
    table(db)
    input("-----Press enter to continue-----")
    menu()

def add():
    while True:
        try:
            Bcode = input("Enter the book code: ")
            if len(Bcode) != 3 or int(Bcode) > 999:
                raise ValueError
            else:
                break
        except:
            print("Book code must be a 3-digit integer")
    while True:
        try:
            Bname = input("Enter the book name: ")
            if len(Bname) > 50:
                raise ValueError
            else:
                break
        except:
            print("Book name can't be longer than 50 chars")
    while True:
        try:
            AuthorName = input("Enter the name of Author: ")
            if len(AuthorName) > 30:
                raise ValueError
            else:
                break
        except:
            print("Author name can't be more than 30 chars")
    while True:
        try:
            Borrower = input("Enter the name of the borrower: ")
            if len(Borrower) > 30:
                raise ValueError
            else:
                break
        except:
            print("Borrower name can't be more than 30 chars")
    while True:
        try:
            IssueDate = input("Enter the date of issuing in appropriate manner as YYYY-MM-DD: ")
            if len(IssueDate) != 10 or IssueDate[4] != "-" or IssueDate[7] != "-":
                raise ValueError
            for i in IssueDate[0:4] + IssueDate[5:7]:
                if i.isdigit() == False:
                    raise ValueError
            break
        except:
            print("IssueDate not written appropriately")
    while True:
        try:
            ReturnDate = input("Enter the date of return in appropriate manner as YYYY-MM-DD: ")
            if len(ReturnDate) != 10 or ReturnDate[4] != "-" or ReturnDate[7] != "-":
                raise ValueError
            for i in ReturnDate[0:4] + ReturnDate[5:7]:
                if i.isdigit() == False:
                    raise ValueError
            break
        except:
            print("ReturnDate not written appropriately")
    cursor.execute(f"Insert into books values({Bcode}, '{Bname}', '{AuthorName}', '{Borrower}', '{IssueDate}', '{ReturnDate}')")
    print("Data entry added")
    input("-----Press enter to continue-----")
    menu()

def delete():
    while True:
        try:
            bcode = input("Enter the code: ")
            cursor.execute(f"delete from books where BCode = {bcode};")
            print("Data entry deleted")
            break
        except:
            print("Couldn't find any book")
    input("-----press enter to continue-----")
    menu()

def update():
    while True:
        try:
            bcode = input("Enter the code: ")
            cursor.execute(f"select * from books where BCode = {bcode};")
            db = cursor.fetchall()
            if len(list(db)) == 0:
                raise ValueError
            break
        except:
            print("Couldn't find any book")
    while True:
        try:
            Borrower = input("Enter new name of the borrower: ")
            if len(Borrower) > 30:
                raise ValueError
            else:
                break
        except:
            print("Borrower name can't be more than 30 chars")
    while True:
        try:
            IssueDate = input("Enter the date of issuing in appropriate manner as YYYY-MM-DD: ")
            if len(IssueDate) != 10 or IssueDate[4] != "-" or IssueDate[7] != "-":
                raise ValueError
            for i in IssueDate[0:4] + IssueDate[5:7]:
                if i.isdigit() == False:
                    raise ValueError
            break
        except:
            print("IssueDate not written appropriately")
    while True:
        try:
            ReturnDate = input("Enter the date of return in appropriate manner as YYYY-MM-DD: ")
            if ReturnDate.lower() == "null":
                break
            elif len(ReturnDate) != 10 or ReturnDate[4] != "-" or ReturnDate[7] != "-":
                raise ValueError
            else:
                for i in ReturnDate[0:4] + ReturnDate[5:7]:
                    if i.isdigit() == False:
                        raise ValueError
            break
        except:
            print("ReturnDate not written appropriately")
    cursor.execute(f"Update books set Borrower = '{Borrower}', IssueDate = '{IssueDate}', ReturnDate = {ReturnDate} where BCode = {bcode};")
    print("Data entry updated")
    input("-----Press enter to continue-----")
    menu()

def find():
    print("How do you wish to find: ")
    print("1 Find by code")
    print("2 Find by name")
    print("3 Find by author")
    print("4 Back")
    print("5 Exit")
    loop2 = True
    while loop2:
        var2 = input("Enter the number: ")
        if var2 == "1":
            loop2 = False
            findbycode()
        elif var2 == "2":
            loop2 = False
            findbyname()
        elif var2 == "3":
            loop2 = False
            findbyauthor()
        elif var2 == "4":
            loop2 = False
            menu()
        elif var2 == "5":
            print("Exitting the application")
            quit()

def table(db, headers=["BCode", "BName", "AuthorName", "Borrower", "IssueDate", "ReturnDate"]):
    print(tabulate(db, headers))

def findbycode():
    while True:
        try:
            bcode = input("Enter the code: ")
            cursor.execute(f"select * from books where BCode = {bcode};")
            db = cursor.fetchall()
            if len(list(db)) == 0:
                raise ValueError
            table(db)
            break
        except:
            print("Couldn't find any book")
    input("-----press enter to continue-----")
    menu()

def findbyauthor():
    while True:
        try:
            aname = input("Enter the name: ")
            cursor.execute(f"select * from books where AuthorName = '{aname}';")
            db = cursor.fetchall()
            if len(list(db)) == 0:
                raise ValueError
            table(db)
            break
        except:
            print("Couldn't find any book")
    input("-----press enter to continue-----")
    menu()

def findbyname():
    while True:
        try:
            bname = input("Enter the name: ")
            cursor.execute(f"select * from books where BName = '{bname}';")
            db = cursor.fetchall()
            if len(list(db)) == 0:
                raise ValueError
            table(db)
            break
        except:
            print("Couldn't find any book")
    input("-----press enter to continue-----")
    menu()

#Main-loop
print("Welcome to the Library Management Software, made by Jaspreet Singh class XII-Confident")
menu()

#Disconnecting
cursor.close()
