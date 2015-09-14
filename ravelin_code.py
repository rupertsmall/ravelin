# ravelin code test

import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime as dt
from datetime import timedelta
from matplotlib import dates
from math import ceil

blob = open('data.csv')
split = csv.reader(blob)
next(split)									# skip first line	

date = []
orders = []
circumstance = []
drivers_available = [];
circ_map = {'very_rainy':'purple', 'rainy':'blue', 'dry':'orange', '':'white'}	# circumstances

# ----------------------------------------------------------------------+
# rules for missing data:						|
# ----------------------------------------------------------------------+
# empty date	=> previous date + 1 day				|
# empty orders  => previous order (not true, but useful for plotting)	|
# circumstances => 0 if not case-insensitive(dry || rainy || very_rainy)|
# drivers	=> previous drivers (not true, but useful for plotting)	|
# ----------------------------------------------------------------------+

# probe + clean using rules
for eachline in split:
	# rules assume first line of date well defined
	
	# date rule
	if eachline[0]:
		date_val = dt.strptime(eachline[0],'%Y-%m-%d')
		date.append(dates.date2num(date_val))
	if not eachline[0]:
		date.append(dates.date2num(date_val + timedelta(days=1)))
	
	# orders rule
	if eachline[1]:	orders_val = eachline[1]; orders.append(float(orders_val)/1000)
	else:	orders.append(float(orders_val)/1000)

	# circumstances rule
	if eachline[2].lower() not in circ_map.keys(): circumstance.append(circ_map[''])
	else: circumstance.append(circ_map[eachline[2].lower()])

	# drivers rule
	if eachline[3]: drivers_val = int(eachline[3]); drivers_available.append(drivers_val)
	else: drivers_available.append(drivers_val)

blob.close()

# # # # # # # # # # # # # # #  PLOTS  # # # # # # # # # # # # # # # # # # # # 

# All Orders data
plt.plot_date(date, orders, fmt='k-')
plt.ylabel("Orders (thousands)")
plt.grid(True)


# A month of data with many indicators
a = 50		# window start date
b = a + 63 	# window end date
av_orders=float(sum(orders))/len(orders)
av_drivers=float(sum(drivers_available))/len(drivers_available)
fig = plt.figure()

# zero normalised orders
orders_section = orders[a:b]
orders_section = np.array(orders_section)*float(len(orders_section))/sum(orders_section) - 1
drivers_section = drivers_available[a:b]
drivers_section = np.array(drivers_section)*float(len(drivers_section))/sum(drivers_section) - 1
date_section = date[a:b]
plt.plot_date(date_section, orders_section, fmt='r-o',ms=10,label='Z.N. Orders')
plt.ylabel("zero normalised units")
plt.grid(True)

# color bars for weather
for i, circ in enumerate(circumstance[a:b]):
        plt.bar(date_section[i], .299, 1, color=circ, linewidth=0)
	plt.bar(date_section[i], -.299, 1, color=circ, linewidth=0)


# normalised drivers - normalised orders
plt.bar(date[a:b], [x[0]-x[1] for x in zip(drivers_section,orders_section)], 1, color='white',label='N.D. - N.O.')
plt.legend()

# Take Fourier Transform to see periodicity
N = len(orders)
spacing = 1
select_arr = range(1,int(ceil(N/20))+1)
k = (1./(N*spacing))*np.linspace(0,(N-1)/2.,N)
Y = np.fft.fft(orders)/N
k = k[select_arr]
Y = Y[select_arr]


fig = plt.figure()
plt.bar(k, abs(Y), .0005, color='blue')
plt.xlabel("freq")
plt.ylabel("power")
plt.grid(True)


# color the peaks in
plt.bar([0.0050955414012738851, 0.0057324840764331206], [0.43387247023994885, 0.37578970902631248], .0005, color='red')
plt.bar([0.0070063694267515917, 0.0076433121019108281], [0.21427122902096354, 0.21438079769271132], .0005, color='yellow')
plt.bar([0.010828025477707006], [0.23553337269496424], .0005, color='green')


# # # # # # # # # # # # # # # # # # # # PRINT SOME STATS # # # # # # # # # # # # # # # # # # # # # # # # #

wet_av=[]
day_av=[]

for k in range(0,7):
	day_orders = [order for i, order in enumerate(orders) if date[i] % 7 == k]
	day_prev = [orders[i-1] for i in range(len(orders)) if date[i] % 7 == k]
	wet_orders = [order for i, order in enumerate(orders) if date[i] % 7 == k and (circumstance[i] == 'purple' or circumstance[i] == 'blue')]
	wet_prev = [orders[i-1] for i, order in enumerate(orders) if date[i] % 7 == k and (circumstance[i] == 'purple' or circumstance[i] == 'blue')]
	day_diff = [x[0]-x[1] for x in zip(day_orders, day_prev)]
	wet_diff = [x[0]-x[1] for x in zip(wet_orders, wet_prev)]
	day_av.append(sum(day_diff)/len(day_diff))
	wet_av.append(sum(wet_diff)/len(wet_diff))

print
print '+---------------------------------------+'
print '| wet average:\t', sum(wet_av)/7.,'\t\t|'
print '| day average:\t', sum(day_av)/7.,'\t|'
print '| wet variance:\t', np.var(wet_av),'\t\t|'
print '| day variance:\t', np.var(day_av),'\t|'
print '+---------------------------------------+'
print

# Find N such that average sum of residuals is minimum
# (but first correct for bad weather bumps before doing linear regression)
corrected_orders = np.array(orders)
correct = [i for i, circ in enumerate(circumstance) if circ == 'purple' or circ == 'blue']
corrected_orders[correct] = corrected_orders[correct] - .5
dates = np.array(date)
L = 31
errors=np.zeros(L-2)
err_N_is_12 = []
predictions = []
for N in range(2,L):
	one = np.ones(N)
	d_select = dates[0:N]
	o_select = corrected_orders[0:N]
	t=0
	for i in range(N+1,len(dates)):
		M = np.c_[d_select, one]
		coeff = np.linalg.lstsq(M,o_select)[0]
		d_select = dates[i-N:i]
		o_select = corrected_orders[i-N:i]
		errors[N-2] = errors[N-2] + (coeff[0]*d_select[-1] + coeff[1] - o_select[-1])**2
		if N==12:
			if circumstance[i] == 'purple' or circumstance[i] == 'blue': bad = 1
			else: bad = 0
			predict = coeff[0]*d_select[-1] + coeff[1] + bad*.5
			predictions.append(predict)
			err_N_is_12.append((predict - orders[i])**2)
 			
		t += 1
	errors[N-2] = errors[N-2]/t

# plot errors
x=range(2,L)
fig = plt.figure()
plt.plot(x, errors, 'r-o')
plt.xlabel("N (size of history)")
plt.ylabel("error")
plt.grid(True)

# plot predictions
fig = plt.figure()
plt.plot_date(date, orders, fmt='k-',label='actual')
plt.plot_date(date[13:], predictions, fmt='r-',label='predicted')
plt.ylabel("Orders (thousands)")
plt.grid(True)
plt.legend()

print 'mean prediction error:\t\t', np.sqrt(np.mean(err_N_is_12))
print 'std dev prediction error:\t', np.sqrt(np.var(err_N_is_12))
print

# plot errors
fig = plt.figure()
plt.plot_date(date[13:], np.sqrt(err_N_is_12), fmt='r-',label='Predictions Error')
plt.ylabel("Predictions Error (Thousands)")
plt.grid(True)
plt.legend()


plt.show()
