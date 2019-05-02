import pyautogui

'''
pyautogui.size()#get your screen size
pyautogui.position()# get the position of your mouse
pyautogui.moveTo()#move your mouse to the position
pyautogui.moveRel()#move the mouse to the screen relevant to current position
pyautogui.click()
pyautogui.doubleClick()
pyautogui.rightClick()
'''

#Drag
pyautogui.click() # click to put drawing program in focus
distance=2000
while distance >0:
    print(distance,0)
    pyautogui.dragRel(distance,0,duration=0.1) #move right
    distance=distance-25

    print(0,distance)
    pyautogui.dragRel(0,distance,duration=0.1) #move down

    print(-distance,0)
    pyautogui.dragRel(-distance,0,duration=0.1) #move left

    distance=distance-25

    print(0,-distance)
    pyautogui.dragRel(0,-distance,duration=0.1) #move up
