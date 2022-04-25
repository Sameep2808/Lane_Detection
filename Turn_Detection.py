import cv2
import os
import matplotlib.pyplot as plt
import math 
import numpy as np
from scipy import stats
from numpy import linalg as LA

def show(img):
	cv2.imshow('Output', img)
	cv2.waitKey()
	cv2.destroyAllWindows()

def showv(img):
	cv2.imshow('Output', img)
	cv2.waitKey(1)

def main():
	cam = cv2.VideoCapture("challenge.mp4")
	#cam = cv2.VideoCapture("whiteline.mp4")
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	vout = cv2.VideoWriter("Output3.mp4", fourcc, 25, (2280,720))
	ret = True
	while ret == True:
		ret,frame = cam.read()
		if ret == False :
			break
		img = frame
		F = detect(img)
		# print(F.shape)
		F = cv2.resize(F, (2280,720), interpolation = cv2.INTER_AREA)
		vout.write(F)
	vout.release()
		# break

def mask(img,ss):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray,(9,9),cv2.BORDER_DEFAULT)
	# showv(gray)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5,5))
	gray = clahe.apply(gray)
	# showv(gray)
	
	r,mask = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)

	kernel = np.ones((7,7), np.uint8)
	ret = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	ret = cv2.erode(mask, kernel) 
	# showv(ret)

	kernel = np.ones((7,7), np.uint8)
	ret = cv2.dilate(ret,kernel,iterations = 1)
	# show(ret)
	return ret


def draw(img,fpts,p,e,col):
	fpts = np.array(fpts)
	m,n = fpts.shape
	# print(m)
	if m > 200:
		fpts = np.vstack((p[:],fpts))
	a1 = np.abs(stats.zscore(fpts[:,0], axis = 0)) < e
	a2 = np.abs(stats.zscore(fpts[:,1], axis = 0)) < e
	c = np.row_stack((a1,a2)).T
		
	fpts = fpts*c
	fpts = fpts[c.all(axis=1)]
	# cv2.polylines(img, [fpts], True, (0,0,255) , 2)
	c = np.polyfit(fpts[:,1],fpts[:,0],2)
	
	dyx = 2*c[0] + c[1]
	d2yx = 2*c[0]
	R = ( (1 + dyx**2 )**(3/2) ) / d2yx

	pts = []
	h,w,co = img.shape
	for x in range(int(h)):
		y = (c[0]*x*x) + (c[1]*x) + c[2]
			
		pts.append([int(y),int(x)])
			
	pts = np.array(pts)	
	
	imgr = cv2.polylines(img, [pts], False, col , 5)

	return pts,imgr,R	


def lines(img,edges):
	global lp
	global rp
	solid_lines = cv2.HoughLinesP(edges,1,np.pi/180,35,None,None,None)
	
	h,w,c = img.shape
	fpts,left,right = [],[],[]
	if solid_lines is not None:
		for i in range(0, len(solid_lines)):
			l = solid_lines[i][0]
			
			if l[0] < w/2:
				left.append([l[0], l[1] ])
			else:
				right.append([l[0], l[1] ])

			if l[2] < w/2:
				left.append([l[2], l[3] ])
			else:
				right.append([l[2], l[3] ])
		edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
		l,edges,RL = draw(edges,left,lp,1.7, (0,0,255))
		r,edges,RR = draw(edges,right,rp,1.7, (0,255,255))
		lp =  np.vstack((lp,l))
		rp =  np.vstack((rp,r))
		rpf = np.flip(r,0)
		fpts = np.vstack((l,rpf))
		h,w,c = edges.shape
		imgm = np.zeros((h,w,c))
		imgr = cv2.fillPoly(img,[fpts],(0,0,255))
		# show(edges)
	return imgr,lp,rp,edges,RL,RR

def ls(pts):
	A = pts[:,0]
	A = A**2
	o = [1]*len(pts[:,0])
	A = np.vstack((A,pts[:,0],o))
	B = pts[:,1]
	X = np.dot(LA.pinv(A.T),B)
	return X

def dr(img,gimg,edges):
	lower = np.array([0,250,0])
	upper = np.array([0,255,0])
	mask = cv2.inRange(gimg, lower, upper)
	kernel = np.ones((30,30), np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)
	mask = cv2.bitwise_not(mask, mask=None)
	br = cv2.bitwise_and(edges, mask)
	# showv(mask)
	return br


def detect(img):
	global lp
	global rp
	up = homo(img)
	h,w,c = up.shape
	edges = mask(up,0)
	imgr,lp,rp,er,RL,RR = lines(up,edges)
	d = back(img,imgr)
	d = cv2.putText(d, "Left Curvature = "+str(RL), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
	d = cv2.putText(d, "Right Curvature = "+str(RR), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
	d = cv2.putText(d, "Average Curvature = "+str((RL+RR)/2), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
	T = 250
	if RL > RR + T:
		d = cv2.putText(d, "TURN RIGHT", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
	elif RR > RL + T:
		d = cv2.putText(d, "TURN LEFT", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
	else:
		d = cv2.putText(d, "GO STRAIGHT", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
	# show(d)
	edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
	final = np.hstack((d,edges,er))
	showv(final)
	return final
	

def homo(img):
	src=np.array([[620,445],[200,670],[1170,670],[725,445]])
	des=np.array([[0,0],[0,720],[500,720],[500,0]]) 
	H,ret=cv2.findHomography(src,des)
	up=cv2.warpPerspective(img,H,(500,720))
	# show(up)
	return up

def back(img,imgr):
	src=np.array([[620,445],[200,670],[1170,670],[725,445]])
	des=np.array([[0,0],[0,720],[500,720],[500,0]]) 
	H,ret=cv2.findHomography(src,des)
	h,w,c = img.shape
	imgr =cv2.warpPerspective(imgr,LA.inv(H),(w,h))
	# show(imgr)
	dest = cv2.addWeighted(img, 1, imgr, 0.5, 0.0)
	# dest = superimpose(img,imgr)
	# show(dest)
	return dest

def superimpose(img,w_timg):

	m = w_timg
	gray = cv2.cvtColor(w_timg, cv2.COLOR_BGR2GRAY)
	r, g = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
	
	invmask = cv2.bitwise_not(g)
	imgc = img.copy()
	imgc[:,:,0] = invmask
	imgc[:,:,1] = invmask
	imgc[:,:,2] = invmask
	final_mask = cv2.bitwise_and(img, imgc)
	st_img = cv2.add(final_mask,m)
	# st_img = cv2.addWeighted(final_mask, 0.5, m, 1, 0.0)
	return st_img


if __name__ == '__main__':
	global lp
	global rp
	lp = np.array([100000,10000])
	rp = np.array([100000,10000])
	main()



# def disparity(img1,img2,window,s):
# 	h1,w1,c = img1.shape 
# 	h2,w2,c = img2.shape 
# 	imgs1 = cv2.resize(img1.copy(), (int(w1/s),int(h1/s)))
# 	imgs2 = cv2.resize(img2.copy(), (int(w2/s),int(h2/s)))

# 	g1 = cv2.cvtColor(imgs1, cv2.COLOR_BGR2GRAY)
# 	g2 = cv2.cvtColor(imgs2, cv2.COLOR_BGR2GRAY)
	
# 	left_array, right_array = g1,g2
# 	left_array = left_array.astype(int)
# 	right_array = right_array.astype(int)
# 	if left_array.shape != right_array.shape:
# 		raise "Left-Right image shape mismatch!"
# 	h, w = left_array.shape
# 	dmap = np.zeros((h, w))

# 	x_new = w - (2 * window)
# 	for y in tqdm(range(window, h-window)):
# 		block_left_array = []
# 		block_right_array = []
# 		for x in range(window, w-window):
# 			block_left = left_array[y:y + window,
# 									x:x + window]
# 			block_left_array.append(block_left.flatten())

# 			block_right = right_array[y:y + window,
# 									x:x + window]
# 			block_right_array.append(block_right.flatten())

# 		block_left_array = np.array(block_left_array)
# 		block_left_array = np.repeat(block_left_array[:, :, np.newaxis], x_new, axis=2)

# 		block_right_array = np.array(block_right_array)
# 		block_right_array = np.repeat(block_right_array[:, :, np.newaxis], x_new, axis=2)
# 		block_right_array = block_right_array.T

# 		abs_diff = np.abs(block_left_array - block_right_array)
# 		sum_abs_diff = np.sum(abs_diff, axis = 1)
# 		idx = np.argmin(sum_abs_diff, axis = 0)
# 		disparity = np.abs(idx - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
# 		dmap[y, 0:x_new] = disparity 


# 	dmapF = np.uint8( dmap * 255 / np.max(dmap) )
# 	show(dmap)
# 	plt.imshow(dmapF, cmap='hot', interpolation='nearest')
# 	plt.savefig("O1.png")
# 	plt.imshow(dmapF, cmap='gray', interpolation='nearest')
# 	plt.savefig("O1g.png")

# 	baseline = 88.39
# 	f = 1758.23

# 	depth = (baseline * f) / (dmap+1e-10)
# 	depth[depth > 100000] = 100000

# 	depth_map = np.uint8(depth * 255 / np.max(depth))
# 	plt.imshow(depth_map, cmap='hot', interpolation='nearest')
# 	plt.savefig("O1d.png")
# 	plt.imshow(depth_map, cmap='gray', interpolation='nearest')
# 	plt.savefig("O1dg.png")
