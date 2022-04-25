import cv2
import os
import matplotlib.pyplot as plt
import math 
import numpy as np

def show(img):
	cv2.imshow('Output', img)
	cv2.waitKey()
	cv2.destroyAllWindows()

def showv(img):
	cv2.imshow('Output', img)
	cv2.waitKey(50)

def main():
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	vout = cv2.VideoWriter("Output2.mp4", fourcc, 25, (960,540))
	cam = cv2.VideoCapture("whiteline.mp4")
	ret = True
	while ret == True:
		ret,frame = cam.read()
		if ret == False :
			break
		img = frame
		F = detect(img)
		# print(F.shape)
		vout.write(F)
		F = cv2.resize(F, (960,540), interpolation = cv2.INTER_AREA)
	vout.release()
	cv2.destroyAllWindows()
		#break

def mask(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	h,w = gray.shape
	m1 = np.zeros((int(h/2)+50,w), dtype="uint8")
	m2 = np.ones((int(h/2)-50,w),  dtype="uint8")
	m = np.vstack((m1,m2*255))
	
	lower = np.array([220,220,220])
	upper = np.array([255,255,255])
	mask = cv2.inRange(img, lower, upper)
	kernel = np.ones((1,1), np.uint8)
	ret = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	kernel = np.ones((5,5), np.uint8)
	ret = cv2.dilate(ret,kernel,iterations = 1)
	br = cv2.bitwise_and(m, ret)
	# show(br)
	return br

def lines(img,edges,c,t):
	if t == 0:
		solid_lines = cv2.HoughLinesP(edges,1,np.pi/180,200,None,200,20)
	else :
		solid_lines = cv2.HoughLinesP(edges,1,np.pi/180,5,None,10,2)
	
	if solid_lines is not None:
		for i in range(0, len(solid_lines)):

			l = solid_lines[i][0]

			m = (l[3] - l[1])/(l[2]-l[0])
			
			if(abs(m) < 0.5) or (abs(m) > 0.8) :
				continue

			cv2.line(img, (l[0], l[1]), (l[2], l[3]), c, 3, cv2.LINE_AA)

	return img

def dr(img,gimg,edges):
	lower = np.array([0,250,0])
	upper = np.array([0,255,0])
	mask = cv2.inRange(gimg, lower, upper)

	kernel = np.ones((15,15), np.uint8)
	mask = cv2.dilate(mask,kernel,iterations = 1)
	mask = cv2.bitwise_not(mask, mask=None)
	# show(mask)
	br = cv2.bitwise_and(edges, mask)
	# show(br)
	return br


def detect(img):
	# show(img)
	edges = mask(img)
	imgr = lines(img,edges,(0,255,0),0)
	edges = dr(img,imgr,edges)
	imgr = lines(img,edges,(0,0,255),1)
	# showv(edges)
	showv(imgr)
	return imgr

if __name__ == '__main__':
	main()
	