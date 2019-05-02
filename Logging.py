import logging
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')


#Why we dont use print instead of using logging.debug?
#Because we can disable all the logging.debug()
logging.disable(logging.CRITICAL)


logging.debug('start of program')

def factorial(n):
    logging.debug('start of factorial %s'%n)
    total =1
    for i in range(1,n+1):
        total *=i
        logging.debug('i is %s,total is %s'%(i,total))
    logging.debug('return the value is %s'%(total))
    return total

print(factorial(5))


'''
There 5 level of logging.debug

logging.debug()
logging.info()
logging.warning()
logging.error()
logging.critical()

'''
