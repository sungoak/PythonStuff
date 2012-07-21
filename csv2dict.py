import csv
# csv2dict.py
#
# This script takes csv-type data files and converts them into a python dictionary with the "first" line of the csv-file read in as the "headers" and, accordingly, as the dictionary "keys." 
# Created 2012Mar12 by BChoi
# Ex.: mydict = csv2dict('/Users/bchoi/Documents/Rearden_Related/Axciom_Related/acxiomSample2.csv');
#
# INPUTS:
#   fname = location and filename of csvfile containing data
# OUTPUTS:
#   mydict = outputted python dictionary containing "hashtable" of csv-data

# ----------------------------- csv2dict.py -------------------------------
def csv2dict(fname):
 #fname = '/Users/bchoi/Documents/Rearden_Related/Axciom_Related/acxiomSample2.csv'
 data = csv.reader(open(fname,"rU"))
 datalist = [] # initialize list
 datalist.extend(data) # extend data (csvtype) to list type
 headers = datalist[0]; # read in first line of data as "headers"
 listlist = [[] for k in range(len(headers))]; # initialze list of lists to STORE data
 for col in range(len(headers)): # loop thru all headers (features)
  rownum = 0;
  for obs in datalist: # loop thru all "rows" of data
   if rownum == 0: # dont read in first line (headers)
    rownum=0;
   else:
    if len(obs) != len(headers): # fill in with blanks if data in row is insufficient
     for k in range(len(obs),len(headers)):
      obs.append('');
      print "added blank for missing column at location " + str(k)
    listlist[col].append(obs[col]); # "append" all rows in data for col in listlist
   rownum+=1;
 # create a dictionary (hashmap) for easier callable data structure
 mydict = {};
 mydict = dict(zip(headers,listlist));
 return mydict;
 
# ----------------------------- csv2str.py -------------------------------
# Reads a delimited file and converts it into an array of strings
# Created 2012May07 by BChoi 
# 
# Ex.: a = csv2dict.csv2str("/Users/bchoi/Desktop/ex5Data/ex5Linx.dat");
def csv2str(fname):
 data = csv.reader(open(fname,"rU"))
 datalist = [] # initialize list
 datalist.extend(data) # extend data (csvtype) to list type
 dataArray = array(datalist).flatten() # flatten() method used like "squeeze" in MATLAB
 return dataArray

# ----------------------------- csv2num.py -------------------------------
# Reads a delimited file and converts it into a numerical array
# Created 2012May07 by BChoi 
# 
# Ex.: a = csv2dict.csv2num("/Users/bchoi/Desktop/ex5Data/ex5Linx.dat");
from numpy import array 
def csv2num(fname):
 data = csv.reader(open(fname,"rU"))
 datalist = [] # initialize list
 datalist.extend(data) # extend data (csvtype) to list type
 dataArray = array(datalist,float).flatten() # flatten() method used like "squeeze" in MATLAB
 return dataArray