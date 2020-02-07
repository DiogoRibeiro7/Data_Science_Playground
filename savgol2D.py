import numpy as np

def savgol2Dcoeff(order=2,npx=5,npy=5):
	v = np.arange(-(npy-1)/2,((npy+1)/2))
	w = np.arange(-(npx-1)/2,((npx+1)/2))
	#Jacobian
	#Calculating the number of elements in each row
	numval = sum(n+1 for n in range(order+1))
	#declaring Jacobian as an empty matrix 
	J = np.empty([0,numval])
	#computing the jacobian Matrix
	for val_w in w:
		for val_v in v:
			Jrow = []
			order_done = []
			for o in range(order+1):
				for j in range(o+1):
					for i in range(o+1):
						Jrow_val = (val_v**i)*(val_w**j)
						if i+j == o and o not in order_done:
							Jrow.append(Jrow_val)
				order_done.append(o)
			#print Jrow
			J = np.vstack((J,Jrow))
	
	JT = np.transpose(J)
	JTJ = np.matmul(JT,J)
	iJTJ = np.linalg.inv(JTJ)
	C = np.matmul(iJTJ,JT)
	return C

def savgol2Dfilt(data2D,order=2,npx=5,npy=5,h=2):
	savgolcoeff = savgol2Dcoeff(order,npx,npy)
	coeff = savgolcoeff[0,:].reshape((npx,npy))
	padx_len = h*(npx-1)/2
	pady_len = h*(npy-1)/2
	half_xwin = padx_len
	half_ywin = pady_len
	data_pad  = np.pad(data2D,((padx_len,padx_len),(pady_len,pady_len)),'constant',constant_values=(0,0))
	data_shape = data2D.shape
	data_pad_shape = data_pad.shape
	new_data = np.zeros(data_shape)
	for i in range(padx_len,data_pad_shape[0]-padx_len):
		for j in range(pady_len,data_pad_shape[1]-pady_len):
			data_window = data_pad[i-half_xwin:i+half_xwin+1,j-half_ywin:j+half_ywin+1]
			new_data[i-padx_len,j-pady_len] = np.sum(np.multiply(data_window,coeff))
	return new_data

