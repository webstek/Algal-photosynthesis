"""
BIOL 111 Photosynthesis Simulation
"""

#We will use a matrix and linear transformation to create the model
#So, we need to import some libraries to accomplish this
import numpy as np
from matplotlib import pyplot as plt
#from scipy import stats
#from matplotlib import colors as clr


##############################################################################
                               #Variables#
##############################################################################
T = 10 #number of trials to complete

#let us define the size of our simulation space, and the number of iterations
n = 25 #dimension
d = 1 #value used to determine the random movement of the particles in a step

t = 100 #number of iterations per trial
tdisp = t #number of trials between showing of graphs
tphot = 5 #number of trials between checking for photosynthesis

p = 0 #dimension of square of red dots p < n
pr = 3 #range of box around (prx,pry) such that O2 --> CO2
prx = n/2 #location of the box x origin (n/2 sets the middle of plot)
pry = n/2 #Location of the box y origin

#Create a matrix for the O2 numbers so it can be recorded
O2_vals = np.zeros((1, T))

print('-------------------------------------------------------------------------')
print("The simulation has {0} particles, {1} trials, and {2} iterations per trial.".format((n+1)**2,T,t))
print('-------------------------------------------------------------------------')
print('')
print('')


##############################################################################
                    #MAIN FUNCTION INITIAL CONDITIONS#
##############################################################################

#The function
def iterate(t,T):
#We will encode the set of positions and velocities for the set of points.
    SP = np.zeros( ( 5,(n+1)**2 ) ) #Simulation Population
    V = np.identity(5) #Velocity Transform
    V[0][2]=1
    V[1][3]=1

#now, we must fill in the matrix space with the set of coordinates for each xy
    for k in range(0,n+1):
        for i in range(0,n+1):
            SP[0][i+(n+1)*k] = i
            SP[1][i+(n+1)*k] = k
#This has set the coordinates that the points start at


#Now we shall give each particle a random x and y velocity.
    for i in range(0,(n+1)**2):
        SP[2][i] = np.random.uniform(-d,d)
        SP[3][i] = np.random.uniform(-d,d)

#Setting initial O2 or CO2
    for k in range(0,p):
        for i in range(0,p):
            SP[4][i+(n+1)*k] = 0


#Set the colour of initial points
#    col_SP = np.where(SP[4]==1, 'r', 'b')

#Plot the starting positions of the molecules
#    plt.grid(True)
#    plt.scatter(SP[0],SP[1], s=8, c=col_SP)
#    plt.xlabel('x-position')
#    plt.ylabel('y-position')
#    plt.title('Initial Position of $O_2$ and $CO_2$ Molecules')
#    plt.show()
    

#We can multiply by a 5x5 matrix on the left to acheive the new positions
    SP1 = np.dot(V,SP)

#Checks to see if O2 --> CO2 should happen
    for i in range(0,(n+1)**2):
        if np.abs(SP1[0][i]-prx)>=0 and np.abs(SP1[0][i]-prx)<=pr and np.abs(SP1[1][i]-pry)>=0 and np.abs(SP1[1][i]-pry)<=pr:
            SP1[4][i]=1
        
#Setting the colours of the data points after the 1st iteration
    col_SP1 = [None] * (n+1)**2
    for i in range(0,(n+1)**2):
        if SP1[4][i]==1:
            col_SP1[i] = (0, 0.7, 1, 1) #colour of the O2
        else:
            col_SP1[i] = (.6, .6, .6, 0.3) #colour of the CO2
##############################################################################
#Plotting the first iteration
#    plt.scatter(SP1[0],SP1[1], s=6, c=col_SP1)
#    plt.grid(True)
#    plt.rc('axes', axisbelow=True)
#    plt.xlabel('x-position')
#    plt.ylabel('y-position')
#    plt.title('Position of $O_2$ and $CO_2$ Molecules after 1 Iteration')
#    plt.show()

#Display Total O2 molecules
#    O2 = np.sum(SP1, axis=1)
#    print(O2[4])


##############################################################################
                       #MAIN FUNCTION ITERATIONS#
##############################################################################
#Print the trial number
    print('--------------------- Start Trial %s --------------------' % T)

#Move the points according to xdot and ydot
    for j in range(2,t+1):
        SP1 = np.dot(V,SP1)

#Function that switchs points between being O2 and CO2 molecules given they are in a region
        if j%tphot==0:
            for i in range(0,(n+1)**2):
                if np.abs(SP1[0][i]-prx)>=0 and np.abs(SP1[0][i]-prx)<=pr and np.abs(SP1[1][i]-pry)>=0 and np.abs(SP1[1][i]-pry)<=pr:
                    SP1[4][i]=1

#Plot the position matrix with colours representing if it is CO2 or O2
        if j%tdisp==0: 
#Setting the colours of the data points
            col_SP1 = [None] * (n+1)**2
            for i in range(0,(n+1)**2):
                if SP1[4][i]==1:
                    col_SP1[i] = (0, 0.7, 1, 1) #colour of the O2
                else:
                    col_SP1[i] = (.6, .6, .6, 0.3) #colour of the CO2

##############################################################################            
#Plotting the actual graph
            plt.scatter(SP1[0],SP1[1], s=6, c=col_SP1) #Plot iterative Graph
            plt.grid(True)
            plt.rc('axes', axisbelow=True)
            plt.xlabel('x-position')
            plt.ylabel('y-position')
            plt.title('Position of $O_2$ and $CO_2$ Molecules after {} Iterations'.format(j))
            plt.show()
            
#Display the number of converted molecules
            O2 = np.sum(SP1, axis=1)
#            print('Number of O2 points:',O2[4])
        
#restricting the boudary of the points
        for k in range(0,(n+1)**2):
            if SP1[0][k]>n:
                SP1[2][k]*= -1
            if SP1[0][k]<0:
                SP1[2][k]*= -1
            if SP1[1][k]>n:
                SP1[3][k]*= -1
            if SP1[1][k]<0:
                SP1[3][k]*= -1
    
        
#Save the final matrix into a matrix
    global O2_vals
    O2_vals[0][T-1] = globals()['O2F{}'.format(T)] = O2[4] #For trial 1 it saves it as SPF1

##############################################################################    
    print('---------------------- End Trial %s ---------------------' % T)
    print('')

##############################################################################
                    #CALL FUNCTIONS AND THEN PLOT GRAPH#
##############################################################################

#Runs the iteration function T number of times
for i in range(1,T+1):
    iterate(t,i) #takes t and T as inputs

#Calculating O2 Mean of the simulation
O2_avg = (np.sum((O2_vals), axis=1)[0])/T #finds the O2 mean of the simulation
#Setting up (xi - mu)^2 part of equation
O2_dlt = np.zeros((1,T)) #Matrix for O2 deltas from mean
for i in range(0,T):
    O2_dlt[0][i] = (O2_vals[0][i]-O2_avg)
O2_dltsqr = np.zeros((1,T))
for i in range(0,T):
    O2_dltsqr[0][i] = O2_dlt[0][i]**2    

#Calculating O2 SD of the simulation
O2_sd = ( (np.sum((O2_dltsqr), axis=1)[0])/(T-1) )**(1/2)

#Write final values
print('')
print('--------------------------------------------------------')
print("Simulation O2 Average: %.2f" % O2_avg) #Simulation Mean
print("Simulation O2 Standard deviation: %.4f" % O2_sd) #Simulation SD
print('--------------------------------------------------------')
print('')


#Creating the matrix that we will plot
O2_plt = np.zeros((2,T))
O2_plt[0] = O2_vals[0]
for i in range(0,T):
    O2_plt[1][i] = i+1
#O2_plt[1] = O2_vals[0]
    
#Creating the points to draw as the average line
AVG = [[O2_avg, O2_avg],[0,T+1]]

#Finally plotting the graph
plt.scatter(O2_plt[1],O2_plt[0], s=8, c=(0,0,1,1)) #Plots #O2 molecules 
plt.plot(AVG[1],AVG[0], c=(0,0,0,0.7), linestyle='--', linewidth = 2, alpha = 0.5) #Plots mean line
plt.grid(True)
plt.rc('axes', axisbelow=True)
plt.xlabel('Trial Number')
plt.ylabel('Number of $O_2$ Molecules')
plt.title('Number of $O_2$ Molecules graphed by Trial Number')
plt.show()

#plt.plot(x, stats.norm.pdf(x, O2_avg, O2_sd))

