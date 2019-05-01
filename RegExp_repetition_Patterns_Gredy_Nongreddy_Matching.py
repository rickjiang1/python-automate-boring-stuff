import re

# ()? will return 0 or 1 time
batRegex=re.compile(r'Bat(wo)?man')
mo=batRegex.search('the adventures of Batman').group()
print(mo)

batRegex=re.compile(r'Bat(wo)?man')
mo=batRegex.search('the adventures of Batwoman').group()
print(mo)


#if we use *, can find 0 or more
batRegex=re.compile(r'Bat(wo)*man')
mo=batRegex.search('the adventures of Batwowowowowoman').group()
print(mo)

#if we use +, can find 1 or more
batRegex=re.compile(r'Bat(wo)+man')
mo=batRegex.search('the adventures of Batwowowowowoman').group()
print(mo)


#if we want to match the *?+.,  we can use \ to escaping
batRegex=re.compile(r'\*\?\+')
mo=batRegex.search('the *?+ of Batwowowowowoman').group()
print(mo)



# the curly braces can match a specific number of times
batRegex=re.compile(r'Bat(wo){3}man')
mo=batRegex.search('the adventures of Batwowowoman').group()
print(mo)
#Greed matching match the longest string possible

#nongreedy matching the match the shortest string possible
