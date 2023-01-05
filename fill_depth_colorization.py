import os
import argparse
import csv
import warnings
warnings.filterwarnings('ignore')

import scipy
import numpy as np
from scipy.sparse.linalg import spsolve
import cv2

from numba import jit
from tqdm import tqdm

def createFolders(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Failed to create folder at ", path)

@jit
def fill_depth_colorization(imgGray=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = imgGray[ii, jj]

					len_ += 1
					nWin += 1

			curVal = imgGray[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

	  		# Now the self-reference (along the diagonal).
			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1  # sum(gvals(1:nWin))

			len_ += 1
			absImgNdx += 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	new_vals = np.reshape(spsolve(A, b), (H, W), 'F')

	denoisedDepthImg = new_vals * maxImgAbsDepth
    
	output = denoisedDepthImg.reshape((H, W)).astype(np.float16)

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput

	return output


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Fill depth image.")
    parser.add_argument("--csv", default="kitti_train.csv", type=str,
                        help="csv file path")
    parser.add_argument("--raw", default="/home/dataset/EH/DataSet/", type=str,
                        help="Raw image file path. ")
    parser.add_argument("--annotate", default="/home/dataset/EH/DataSet/", type=str,
                        help="Annotate image file path")
    parser.add_argument("--output", default="/home/dataset/EH/DataSet/kitti/annotated_impainting/", type=str,
                        help="Output image file path")
    args = parser.parse_args()
    
    csv_file = []
    with open(args.csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            csv_file.append(line)
            
    # csv_file = np.array(csv_file[:10])
    
    for iter, raw, depth in tqdm(csv_file):
        raw_path = os.path.join(args.raw, raw)
        depth_path = os.path.join(args.annotate, depth)
        
        raw_image = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        output = fill_depth_colorization(imgGray=raw_image, imgDepthInput=depth_image)
        output = np.array(output, dtype=np.uint8)
        
        output_path = os.path.join(args.output,depth[16:])
        createFolders(output_path[:-15])
        cv2.imwrite(output_path, output)