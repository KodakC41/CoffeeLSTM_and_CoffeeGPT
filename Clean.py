out = ""
with open('James.txt', 'r') as f:
    count = 0
    for line in f:
        count+=1
        if count % 2 == 1: #this is the remainder operator
            out+=line
f = open("James_Clean.txt", "w")
f.write(out)
f.close()
out = ""
with open('James_Clean.txt', 'r') as f:
    count = 0
    for line in f:
        count+=1
        if count % 8 == 1: #this is the remainder operator
            out+=line
f = open("James_Clean_Simple.txt", "w")
f.write(out)
f.close()