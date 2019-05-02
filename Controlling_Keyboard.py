import pyautogui



#print(pyautogui.position())
#pyautogui.typewrite('Hw')

pyautogui.click(x=200,y=200)
pyautogui.typewrite("hello world")
pyautogui.typewrite(['Enter'])
pyautogui.typewrite('My Name is:')
pyautogui.typewrite(['Enter'])
pyautogui.typewrite(['J','I','A','N','G','Enter'])
pyautogui.typewrite('Jin Tao')

#check the keyBOARD_Keys
print(pyautogui.KEYBOARD_KEYS)




'''
pyautogui.typewrite()  # can be passed a string of characters to type
pyautogui.press('F1')  #will press a single keyBOARD_Keys
pyautogui.hotkey('ctrl','O')     #can be used for key board shortcuts, like Ctrl+O
'''
