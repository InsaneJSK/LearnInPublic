"""Given an integer 'n'. Print all the possible pairs of 'n' balanced parentheses.
The output strings should be printed in the sorted order considering '(' has higher value than ')'.


Input Format
Single line containing an integral value 'n'.

Output Format
Print the balanced parentheses strings with every possible solution on new line."""

n = int(input())

def genpar(n, openB=0, closeB=0, s=[]):
    if 1<=n<=11:    
        if closeB == n:
            print(''.join(s))
        else:
            if openB > closeB:
                s.append(")")
                genpar(n, openB, closeB+1, s)
                s.pop()
            if openB < n:
                s.append("(")
                genpar(n, openB+1, closeB, s)
                s.pop()
genpar(n)