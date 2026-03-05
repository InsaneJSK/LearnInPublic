#Cout << "Hello World"

class Ostream():
    def __lshift__(self, text):
        print(text, end="")

cout = Ostream()
cout << "JSK"
