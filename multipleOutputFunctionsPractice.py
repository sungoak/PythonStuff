# multiple-returns.py
a,b,c = 0,0,0
def getabc():
	a = "Hello"
	b = "World"
	c = "!"
	return a,b,c	# defines a tuple on the fly

def gettuple():
	a,b,c=1,2,3		# notice similarities between this and getabc()
	return (a,b,c)

def getlist():
	a,b,c = (3,4),(4,5),(5,6)
	return [a,b,c]


