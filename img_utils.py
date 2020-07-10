from global_imports import *


''' wrapper for commonly-used matplotlib display 
	function for debugging '''
def show_img(img, cmap="gray"):
	if len(img.shape) > 2:
		b, g, r = cv2.split(img)
		img_mpl = cv2.merge([r, g, b])
	else:
		img_mpl = img
	plt.axis("off")
	plt.imshow(img_mpl, cmap)
	plt.show()


''' quickly apply a function to an image via a lookup table
	(lookup implementation is in c++) '''
def apply_lut_fn(img, fn):
	
	lookUpTable = np.empty((1,256), np.uint8)
	for i in range(256):
		lookUpTable[0,i] = np.clip(fn(i), 0, 255)
	
	return cv2.LUT(img, lookUpTable)


''' brighten using gamma value (exponential function) '''
def gamma_brighten(img, g=0.2):
	return apply_lut_fn(img, lambda x: np.clip(pow(x / 256.0, g) * 256.0, 0, 255))    


''' repeated-use opencv drawing code '''
def cv2_cross(img, pos, size, colour, t=1):
	y, x = [int(i) for i in pos]
	cv2.line(img, tuple([y - size, x - size]), tuple([y + size, x + size]), colour, t)
	cv2.line(img, tuple([y + size, x - size]), tuple([y - size, x + size]), colour, t)

def cv2_text(image, txt, pos, colour, scale=0.5, thickness=1):
	pos = tuple([int(i) for i in pos])

	font = cv2.FONT_HERSHEY_SIMPLEX 
	fontScale = scale    
	thickness = thickness

	return cv2.putText(image, txt, pos, font,  
					   fontScale, colour, thickness, cv2.LINE_AA) 


''' standard image format is ubytes for pixel intensities.
	if used on an array (eg to rescale deltas from mean),
	this will use the maximum and minimum from the whole array. '''
def rescale_to_ubyte(img):
	img = img.astype(float)
	img -= img.min()
	img /= img.max()
	img *= 255
	return img.astype(np.uint8)


''' ease-of-use function to keep in ubyte type '''
def mean_img(imgs):
	return np.round(np.mean(imgs, axis=0)).astype(np.uint8)


''' given base image, array of contours, and array of scores
	corresponding to the contours (zero to some max val), show 
	an image with the contours drawn and colourmapped by score.'''
def draw_colourmapped_contours(img, contours, scores):
	if len(img.shape) > 2:
		contoured = img.copy()
	else:
		contoured = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	scores = np.array(scores)
	max_val = max(scores)
	scores /= max_val
	colours = cm.winter(scores)
	colours = colours[:,1:]*256

	for i, contour in enumerate(contours):
		contoured = cv2.drawContours(contoured, contours, i, colours[i], 1)

	show_img(contoured)