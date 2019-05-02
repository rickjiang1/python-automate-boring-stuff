import pyautogui
import os

#take screenshot and store in the folder
'''
pyautogui.screenshot()
pyautogui.screenshot('C:\\PythonProject\\automate_boring_stuff\\Image\\test.png')
'''


# important : Need to take a perfect screenshot, otherwise wont be recognize
print(pyautogui.locateOnScreen('C:\\PythonProject\\automate_boring_stuff\\Image\\7.png'))
print(pyautogui.locateCenterOnScreen('C:\\PythonProject\\automate_boring_stuff\\Image\\7.png'))

pyautogui.moveTo(x=2155,y=774)
pyautogui.click(x=2155,y=774)
