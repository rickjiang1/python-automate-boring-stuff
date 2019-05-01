'''
def div42by(diviby):
    return 42/diviby

print(div42by(2))
print(div42by(12))
print(div42by(0))
print(div42by(1))
'''
#use the try except the skip the failed one
'''
def div42by(diviby):
    try:
        return 42/diviby
    except ZeroDivisionError:
        print('Error: you tried to devide by zero')
print(div42by(2))
print(div42by(12))
print(div42by(0))
print(div42by(1))
'''

#if you dont specified the ZeroDivisionError, it will catch all errors
print('How many cats do you have ?')
numCats=input()
try:
    if int(numCats)>=4:
        print('that is a lot of cats')
    else:
        print('That is not that many cats')
except:
    print('please give a number')
