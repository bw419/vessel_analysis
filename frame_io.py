from global_imports import *
from img_utils import *
import os


FRAME_FILENAME_TEMPLATE = "hvi-video-{:05d}"

INPUT_FOLDER = None
INPUT_EXTENSION = None


''' read an individual frame.
	- file name structure given by FRAME_FILENAME_TEMPLATE.
	- folder path and file extension need to be provided only
	  once in the program. '''
def read_frame_i(i, input_folder=None, input_extension=None):

	global INPUT_FOLDER
	global INPUT_EXTENSION

	if input_folder is None or input_extension is None:
		input_folder = INPUT_FOLDER
		input_extension = INPUT_EXTENSION
	if input_folder is None or input_extension is None:
		raise Exception("No file path has been provided.")

	INPUT_FOLDER = input_folder
	INPUT_EXTENSION = input_extension
	frame_path = input_folder + "/" + FRAME_FILENAME_TEMPLATE.format(i) + "." + input_extension
	try:
		return cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)
	except:
		raise Exception("Failed to read image.")


''' read a range of frames from start_n to end_n-1 inclusive.'''
def get_frame_range_from_folder(input_folder, input_extension, start_n, end_n):
	return np.array([read_frame_i(i, input_folder, input_extension) for i in range(start_n, end_n)])


''' read all frames from a folder.
	assumes the folder only contains ordered images of same res.'''
def get_frames_from_folder(input_folder):
	filepaths = os.listdir(input_folder)
	try:
		return np.array([cv2.cvtColor(cv2.imread(input_folder + "/" + fp), cv2.COLOR_BGR2GRAY) for fp in filepaths])
	except:
		raise Exception("Error attempting to read images.")



''' write frames to video given by out_path.
	currently unable to do lossless compression. '''
def write_video_from_frames(frames, out_path):

	codec = 'MJPG'

	height, width = frames[0].shape
	res = (width, height)

	print("writing "  + out_path + "...")
	out_vid = cv2.VideoWriter(out_path, 
							  cv2.VideoWriter_fourcc(*codec),
							  30, res)

	for i in range(len(frames)):
		out_vid.write(cv2.cvtColor(rescale_to_ubyte(frames[i]), cv2.COLOR_GRAY2BGR))
			
	out_vid.release()
	print("done.")


'''	save frames into folder at folder_path.
	creates this folder if it doesn't exist.
	filenames are according to FRAME_FILENAME_TEMPLATE.
	frame numbers are sequential, starting from start_n(=0). '''
def save_frames_to_folder(frames, folder_path, start_n=0):

	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	print("writing "  + folder_path + "...")
	for i in range(len(frames)):
		path = folder_path + "/" + FRAME_FILENAME_TEMPLATE.format(start_n + i) + ".bmp"
		cv2.imwrite(path, frames[i])
	print("done.")


'''	for the purpose of testing stabilisation 'live'. '''
def sim_async_input(frames, framerate, print_timing=False):
	time_delta = 1e9 / framerate
	s_time = time.time_ns()

	n_frames = 0
	overall_start_time = time.time()
	overall_leeway_time = 0
	min_leeway_time = math.inf


	for i, frame in enumerate(frames):
		yield frame

		leeway_time = (s_time + (i+1)*time_delta)-time.time_ns()

		if print_timing:
			overall_leeway_time += leeway_time
			min_leeway_time = min(leeway_time, min_leeway_time)
			n_frames = i+1

		while leeway_time > 0:
			leeway_time = (s_time + (i+1)*time_delta)-time.time_ns()
			continue


	if print_timing:

		overall_end_time = time.time()
		overall_per_frame = (overall_end_time - overall_start_time)/n_frames
		mean_leeway_time = overall_leeway_time/n_frames

		print("mean time processing between frames: {:.1f}%".format(100*(1-mean_leeway_time/time_delta)))
		print("max time processing between frames: {:.1f}%".format(100*(1-min_leeway_time/time_delta)))
		print("actual final framerate: {:.3f} fps".format(1/overall_per_frame))
