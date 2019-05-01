#regular expression basics


#this is the way of how to define a phonenumber
'''
def isPhoneNumber(text):
    if len(text)!=12:
        print('This is more than 12 digit')
        return False

    for i in range(0,3):
        if not text[i].isdecimal():
            return False #no area code
    if text[3]!='-':
        return False
    for i in range(4,7):
        if not text[i].isdecimal():
            return False
    if text[7] !='-':
        return False #missing second dash
    for i in range(8,12):
        if not text[i].isdecimal():
            return False
    return True
print(isPhoneNumber('501-827-0039'))
'''
#use regular expressions
message='Call me 415-555-1011 tommorrow, or at 501-827-0039'
foundNumber=False

import re
PhoneNUmberRe=re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

#find one
print(PhoneNUmberRe.search(message).group())
print('\n')
#find all
print(PhoneNUmberRe.findall(message))
