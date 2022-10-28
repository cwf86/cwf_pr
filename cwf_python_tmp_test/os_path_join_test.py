import os

a="/abc/3rt"
b="/tag/3456" # 因为/开头，所以会被join从/截断

if b[0] == "/":
    b=b[1:]

c=os.path.join(a,b)
print(c)