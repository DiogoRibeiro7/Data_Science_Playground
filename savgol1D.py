import numpy as np
import time
def savgol1Dcoeff(order=2,nump=5):
	z = np.arange(-(nump-1)/2,(nump+1)/2).reshape(-1,1)
	J = np.hstack(np.power(z,n) for n in xrange(order+1))
	JT = np.transpose(J)
	JTJ = np.matmul(JT,J)
	iJTJ = np.linalg.inv(JTJ)
	C = np.matmul(iJTJ,JT) #iJTJ_JT
	return C

def savgolfilt(data,order=2,nump=5,h=2):
	savgolcoeff = savgol1Dcoeff(order,nump)
	coeff = savgolcoeff[0,:]
	print (coeff)
	pad_length = h*(nump-1)/2
	half_window = pad_length
	data_pad  = np.pad(data,(pad_length,pad_length),'constant',constant_values=(0,0))
	data_len = len(data)
	data_pad_len = len(data_pad)
	new_data = np.zeros(data_len)
	for i in range(pad_length,data_pad_len-pad_length):
		data_window = data_pad[i-half_window:i+half_window+1]
		data_smooth = data_window[[h*n for n in xrange(nump)]]
		new_data[i-pad_length] = np.sum(np.multiply(data_smooth,coeff))
	return new_data

