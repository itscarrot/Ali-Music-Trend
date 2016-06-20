import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import os
import datetime

def get_data(file_name):
	data = pd.read_csv(file_name)
	X_parameter = []
	Y_parameter = []
	date_play = {}
	for date,weekday,week_of_month,day_of_month,is_festival,is_holiday,workingday,avg,yesterday_play,seven_day_avg,last_month_avg,play in zip(data['date'],data['weekday'],data['week_of_month'],data['day_of_month'],data['is_festival'],data['is_holiday'],data['workingday'],data['avg'],data['yesterday_play'],data['seven_day_avg'],data['last_month_avg'],data['play']):
	   X_parameter.append([weekday,week_of_month,day_of_month,is_festival,yesterday_play,seven_day_avg,last_month_avg])
	   Y_parameter.append(play)
	   date_play.setdefault(str(date),play)
	return X_parameter,Y_parameter,date_play


def random_forest_model_main(X_parameters,Y_parameters,predict_value):
	model = RandomForestRegressor(n_estimators=100)
	model.fit(X_parameters,Y_parameters)
	predict_outcome = model.predict(predict_value)
	return predict_outcome	

def GBR_model_main(X_parameters,Y_parameters,predict_value):
	model = GradientBoostingRegressor()
	model.fit(X_parameters,Y_parameters)
	predict_outcome = model.predict(predict_value)
	return predict_outcome	

def linear_model_main(X_parameters,Y_parameters,predict_value):
	regr = linear_model.LinearRegression()
	regr.fit(X_parameters, Y_parameters)
	predict_outcome = regr.predict(predict_value)
	predictions = {}
	predictions['intercept'] = regr.intercept_
	predictions['coefficient'] = regr.coef_
	predictions['predicted_value'] = predict_outcome
	return predictions

def get_test_data(file_name):
	data = pd.read_csv(file_name) 
	P_parameter = []
	for date,weekday,week_of_month,day_of_month,is_festival,is_holiday,workingday,avg in zip(data['date'],data['weekday'],data['week_of_month'],data['day_of_month'],data['is_festival'],data['is_holiday'],data['workingday'],data['avg']):
		   P_parameter.append([weekday,week_of_month,day_of_month,is_festival])
	return P_parameter

def get_test_data1(file_name):
	data = pd.read_csv(file_name) 
	P_parameter = []
	# R_parameter = []
	for date,weekday,week_of_month,day_of_month,is_festival,is_holiday,workingday,avg in zip(data['date'],data['weekday'],data['week_of_month'],data['day_of_month'],data['is_festival'],data['is_holiday'],data['workingday'],data['avg']):
		   P_parameter.append([date,weekday,week_of_month,day_of_month,is_festival,is_holiday,workingday,avg])		   
	return P_parameter



oo= os.listdir("./new_train_data/")
for fileName in oo:
	if "DS_Store" in fileName:
		continue

	singerName = fileName.split('.')[0]
	print singerName
	P = get_test_data("./new_predict_data/"+singerName+".csv")
	P1 = get_test_data1("./new_predict_data/"+singerName+".csv")
	output = open("./p2_whole_result.csv", 'a')

	for pdata,tdata in zip(P,P1):
		X,Y,D = get_data("./new_train_data/"+singerName+".csv")

		d1 = datetime.datetime.strptime(str(tdata[0]),'%Y%m%d')  
		d2 = d1+datetime.timedelta(-1)
		
		yesterday_play = D.get(d2.strftime('%Y%m%d'))

		last_7_days_play = 0
		for i in range(1, 30 + 1):
			current_date = d1+datetime.timedelta(-i)
			last_7_days_play = last_7_days_play+ D.get(current_date.strftime('%Y%m%d'))

		seven_avg=	last_7_days_play/7

		pdata.append(yesterday_play)
		pdata.append(seven_avg)
		last_monnth_play_count = 0
		for i in range(1, 30 + 1):
			current_date = d1+datetime.timedelta(-i)
			last_monnth_play_count = last_monnth_play_count+ D.get(current_date.strftime('%Y%m%d'))

		last_month_avg=	last_monnth_play_count/30	
		pdata.append(last_month_avg)


		result = random_forest_model_main(X,Y,pdata)
		
		# result = GBR_model_main(X,Y,pdata)
		# mode_result = linear_model_main(X,Y,pdata)
		# result = mode_result['predicted_value']
		print "Predicted value: ",str(result).strip('[]')

		predict_play = str(result).strip('[]').replace(" ","")

		
		train_file = open("./new_train_data/"+singerName+".csv", 'a')
		train_file.write('\n'+singerName+","+str(tdata[0])+","+str(tdata[1])+","+str(tdata[2])+","+str(tdata[3])+","+str(tdata[4])+","+str(tdata[5])+","+str(tdata[6])+","+str(predict_play)+","+str(tdata[7])+","+str(yesterday_play)+","+str(seven_avg)+","+str(last_month_avg))
		train_file.close()
		if(str(tdata[0])!="20150831"):
			output.write(singerName+","+str(long(float(predict_play)))+","+str(tdata[0])+'\n')

	output.close()
