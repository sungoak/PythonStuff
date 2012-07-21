# sequentialFill.py
# This function "fills" in missing and skipped values of an array or list
# in a sequential manner. This is particularly useful for FFT analysis.
# Created 2011Dec21 by BChoi
# INPUTS:
#   sig = current signal containing missing or skipped values (non-sequential)
#   t = time vector of sig data (with missing or skipped values)
#   targetLen = desired length of target sequential signal
# OUTPUTS:
#   t3 = modified target vector 
# Example: 
# t = dictInstance(mydict,'airportCode',airport,'hour');        # hour, time variables
# x = dictInstance(mydict,'airportCode',airport,'numDelDep');   # num of delayed departing flights
# seqSig = sequential(t);
# t_seq = sequentialFill(t,t,24*365);
# x_seq = sequentialFill(x,t,24*365);
from numpy import *
from pylab import * # imports is_string_like function
from strman import strcmplist
def sequentialFill(sig,t,targetLen):
 # determine if we are returning a time vector or an actual signal vector to modify
 if sig == t: # case where we are sequentially filling time vector with NO ZEROS!
  print "Returning sequentially filled time vector with no zeros!"
  returnType = 0;
 else: # case where we are sequentially filling signal vector with Zeros for missing values!
  print "Returning sequentially filled signal vector with zeros for missing values!"
  returnType = 1;
 # need to create "interpolating" function to "fill" in "0's" where there are no delayed flights!
 # this is needed for proper time alignment and for better FFT results comparison
 #t = double(t);             # convert 't' to double array type
 tsig = range(targetLen);   # sequential vector containing desired or target number of points
 lenCurr = len(t);          # number of time points for current airportCode
 firstCurr = int(t[0]);     # first time point of current airportCodes
 # check to make sure "time" index begins at 0 or 1
 if max(double(t)) > len(double(t)): 
  t = list(double(t) - min(double(t))*ones_like(double(t))); # returns t adj to start from "1" of len(t)
  print "time vector shifted to begin at '0'"
 # Check for time vector TYPE for comparison:
 if is_string_like(t[0]):
  #locOfZero = nonzero(t==min(double(t)));  # find location of starting value in time vector
  locOfZero = strcmplist(t,str(int(min(double(t)))));
 else:
  locOfZero = find(double(t) == 0);
 # Adjust "time" vector to be sequentially ordered
 t2 = t[locOfZero[0]:] + t[0:locOfZero[0]];
 t2 = double(t2);   # convert "strings" to doubles, then convert to LIST type
 t2 = list(t2);
 t2diff = diff(t2);
 t2diffLoc = nonzero(t2diff > 1)[0]; # find all instances where "diff" > 1 (non-sequential!)
 # Now take the difference information and "insert" zeros in numbers reflecting "diff" values
 loopcount = 1;  # initialize boolean counter for first statement in for-loop
 for k in range(len(t2diffLoc)):    # loop thru all non-sequential portions in time vector
  k = int(k);   # convert k from double to int, since t2diffLoc is a double type array
  print "Percentage complete of for-loop: " + str(double(k+1)/len(t2diffLoc))
  if loopcount == 1:
   #t3 = t2[0:t2diffLoc[k]+1] + list(t2diffLoc[k]+1:t2diff[t2diffLoc[k]]-1));  # add "zeros" for missing values
   print t2diffLoc[k]+1
   #t3 = t2[0:t2diffLoc[k]+1] + list(range(int(t2diffLoc[k]+1),int(t2diffLoc[k]+t2diff[t2diffLoc[k]])));
   #t3 = t2[0:t2diffLoc[k]+1] + list(range(int(t2diffLoc[k]+1),int(t2diffLoc[k+1])));
   t3 = t2[0:t2diffLoc[k]] + list(range(int(t2[int(t2diffLoc[k])]),int(t2[int(t2diffLoc[k]+1)])));
   mod_sig = sig[0:t2diffLoc[k]+1] + list(zeros(t2diff[t2diffLoc[k]]-1));
   print "For k = " + str(k) + ", Length of t3 is now: " + str(len(t3));
   loopcount = 0;
  else: #loopcount == 1 & k < len(t2diffLoc)-1: # while NOT last element!
   #t3 = t3 + t2[t2diffLoc[k-1]+1:t2diffLoc[k]+1] + list(zeros(t2diff[t2diffLoc[k]]-1));
   print t2diffLoc[k]+1
   #t3 = t3 + t2[t2diffLoc[k-1]+1:t2diffLoc[k]+1] + list(range(int(t2diffLoc[k]+1),int(t2diffLoc[k]+t2diff[t2diffLoc[k]]))); 
   #t3 = t3 + t2[t2diffLoc[k-1]+1:t2diffLoc[k]+1] + list(range(int(t2diffLoc[k]+1),int(t2diffLoc[k+1])));
   t3 = t3 + list(range(int(t2[int(t2diffLoc[k-1]+1)]),int(t2[int(t2diffLoc[k]+1)])));
   mod_sig = mod_sig + sig[t2diffLoc[k-1]+1:t2diffLoc[k]+1] + list(zeros(t2diff[t2diffLoc[k]]-1));
   print "For k = " + str(k) + ", Length of t3 is now: " + str(len(t3));
   loopcount = 0;
  #else:
   #t3 = t3 + t2[t2diffLoc[k-1]+1:t2diffLoc[k]+1] + list(zeros(t2diff[t2diffLoc[k]]-1));
   #print t2diffLoc[k]+1
   #print type(int(t2diffLoc[k]+1))
   #print type(int(t2diffLoc[k]+t2diff[t2diffLoc[k]]))
   #print range(int(t2diffLoc[k]+1),int(targetLen))
   #t3 = t3 + t2[t2diffLoc[k-1]+1:t2diffLoc[k]+1] + list(range(int(t2diffLoc[k]+1),int(t2diffLoc[k]+t2diff[t2diffLoc[k]]))); 
   #t3 = t3 + t2[t2diffLoc[k-1]+1:t2diffLoc[k]+1] + list(range(int(t2diffLoc[k]+1),targetLen));
   #t3 = t3 + list(range(int(t2diffLoc[k]+1),targetLen));
   #mod_sig = mod_sig + sig[t2diffLoc[k-1]+1:t2diffLoc[k]+1] + list(zeros(t2diff[t2diffLoc[k]]-1));
   #print "For k = " + str(k) + ", Length of t3 is now: " + str(len(t3));
   #loopcount = 0;
 # After accounting for "missing" values, need to append with "last values"
 shortage = len(tsig) - len(t3);
 t3 = t3 + list(zeros(shortage));
 mod_sig = mod_sig + list(zeros(shortage));
 # we need to "re-cat" the desiredTime array to account for staggered time points of flights data
 t3 = t3[t3.index(firstCurr):] + t3[0:t3.index(firstCurr)];
 mod_sig = mod_sig[t3.index(firstCurr):] + mod_sig[0:t3.index(firstCurr)];
 if returnType == 0: # case where we are sequentially filling time vector with NO ZEROS!
  print "Returning sequentially filled time vector with no zeros!"
  return t3;
 else: # case where we are sequentially filling signal vector with Zeros for missing values!
  print "Returning sequentially filled signal vector with zeros for missing values!"
  return mod_sig;
 
