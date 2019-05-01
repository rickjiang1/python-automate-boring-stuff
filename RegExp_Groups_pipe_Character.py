#Group
import re
phoneNumRegx=re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')
mo=phoneNumRegx.search('My number is 501-827-0039')
print(mo.group())


phoneNumRegx=re.compile(r'(\d\d\d)-(\d\d\d-\d\d\d\d)')
mo=phoneNumRegx.search('My number is 501-827-0039')
print('group 1 :'+mo.group(1))

phoneNumRegx=re.compile(r'(\d\d\d)-(\d\d\d-\d\d\d\d)')
mo=phoneNumRegx.search('My number is 501-827-0039')
print('group 2 :'+mo.group(2))


# | pip can match one of many possible groups
batRegx=re.compile(r'Bat(man|mobile|copter|bat)')
mo=batRegx.search('Batmobile lost a wheel')
print('\n')
print(mo.group())
