from global_imports import *
from img_utils import *
from maths_utils import *
import networkx as nx
import thinning


''' create an image to display from a graph object.
	- img: base image to draw the rest on
	- colorcode: contour colour depends on its score
	- dilate: make nodes slightly larger '''
def graph_to_img(G, img, ordered=False, approxed=False, dilate=False, colourcode=False):
		if len(img.shape) < 3:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

		for node in G.nodes:
			t = G.nodes[node]["type"]
			cnt = G.nodes[node]["contour"]
			if t == "joint":
				cv2.drawContours(img, [cnt], -1, (255, 0, 0), 1)
			if t == "end":
				cv2.drawContours(img, [cnt], -1, (255, 0, 0), 1)

		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
		if dilate:
			img = cv2.dilate(img, kernel, iterations=1)



		for u, v, w in G.edges:
			if approxed:
				cnt = G[u][v][w]["point chain"]
				ordered = True
			else:
				cnt =  G[u][v][w]["contour"]
			if colourcode:
				if ordered:
					cv2.polylines(img, [cnt], False, (0, int(126 + 0.5*G[u][v][w]["score"]), int(255 - 0.5*G[u][v][0]["score"])), 1)
				else:
					cv2.drawContours(img, [cnt], -1, (0, int(126 + 0.5*G[u][v][w]["score"]), int(255 - 0.5*G[u][v][0]["score"])), 1)
			else:
				if ordered:
					cv2.polylines(img, [cnt], False,  (0, 255, 0), 1)
				else:
					cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)

		return img


''' create an image to display from skeleton features.
	- img: base image to draw the rest on'''

def features_to_img(img, joints, ends, edges):
	if len(img.shape) < 3:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	img[:,:,0][edges!=0]   = edges[edges!=0]
	img[:,:,1:][edges!=0]  = 0
	img[:,:,1][ends!=0]    = ends[ends!=0]
	img[:,:,0::2][ends!=0] = 0
	img[:,:,2][joints!=0]  = joints[joints!=0]
	img[:,:,:2][joints!=0]  = 0
	
	return img


''' apply Guo-Hall thinning algorithm. 
	- remove_borders: if True, removes border artefacts on initial
	  skeletonisation by setting all border values to 0. '''
def skeletonisation(img, remove_borders=False):

	skeleton = thinning.guo_hall_thinning(img.copy())

	if remove_borders:
		skeleton[0,:] = 0
		skeleton[:,0] = 0
		skeleton[-1,:] = 0
		skeleton[:,-1] = 0

	return skeleton


''' utility functions to complete the set. '''
def skel_to_graph(skel):
	return features_to_graph(*skel_to_features(skel))

def graph_to_skel(G, shape, simplify=False):
	return features_to_skel(*graph_to_features(G, shape), simplify)


''' combine joints, ends and edges.
	- simplify: it True, dilate nodes and then entire skeleton before 
	  re-thinning to get rid of mini spurs and help connect gaps. '''
def features_to_skel(joints, ends, edges, simplify=False):

	if not simplify:
		return joints + ends + edges

	else:
		# dilate to prevent extra mini spurs and help connect gaps
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		dilated = cv2.dilate(joints + ends, kernel, iterations=1)
		edges_no_overlap = edges.copy()
		edges_no_overlap[dilated > 0] = 0
		skel = cv2.dilate(dilated + edges_no_overlap, kernel, iterations=1)
		
		return thinning.guo_hall_thinning(skel)


''' find the morphological skeleton features:
	do so via convolution for speed (as opencv operations are in c):
	convolve with 3x3 kernel of all ones and a 10 in the middle. The
	10 allows us to only consider output pixels > 10 so that only white
	pixel centres are considered.
	- single pixels (islands) are removed.
	- pixels connected to 1 other are considered ends (thresh: >10 but not >11)
	- pixels connected to 2 others are considered edges (thresh: >11 but not >12)
	- pixels connected to 3+ others are considered joints (thresh: >12). '''
def skel_to_features(skel):
	
	kernel = np.ones((3,3), np.uint8)
	kernel[1,1] = 10

	ret, skel_all_ones = cv2.threshold(skel,1,1,cv2.THRESH_BINARY)
	n_adjacents = cv2.filter2D(skel_all_ones, -1, kernel)

	# remove black pixel centres and all pixels not at least 1-connected.
	ret, gt_10 = cv2.threshold(n_adjacents,10,255,cv2.THRESH_BINARY)
	ret, gt_11 = cv2.threshold(n_adjacents,11,255,cv2.THRESH_BINARY)
	ret, gt_12 = cv2.threshold(n_adjacents,12,255,cv2.THRESH_BINARY)
	
	# true==255, false==0 here
	joints = gt_12
	ends = cv2.bitwise_xor(gt_11, gt_10)
	edges = cv2.bitwise_xor(gt_12, gt_11)

	joints = np.bitwise_and(skel.copy(), joints)
	ends = np.bitwise_and(skel.copy(), ends)
	edges = np.bitwise_and(skel.copy(), edges)

	return joints, ends, edges


''' Draw joints, ends, and edges from a graph object. '''
def graph_to_features(G, shape):

	joints = np.zeros(shape, dtype=np.uint8)
	ends = joints.copy()
	edges = joints.copy()


	img = cv2.cvtColor(joints, cv2.COLOR_GRAY2BGR)

	for node in G.nodes:
		t = G.nodes[node]["type"]
		cnt = G.nodes[node]["contour"]
		s = int(G.nodes[node]["score"])
		if t == "joint":
			cv2.drawContours(joints, [cnt], -1, s, 1)
		if t == "end":
			cv2.drawContours(ends, [cnt], -1, s, 1)

	for u, v, w in G.edges:
		cv2.drawContours(edges, [G[u][v][w]["contour"]], -1, int(G[u][v][w]["score"]), 1)

	return joints, ends, edges


deltas = [-1, 0, 1]
xy_deltas = [np.array([x, y]) for x in deltas for y in deltas]
del xy_deltas[4]
''' construct graph object from joints, ends, and edges. 
	this involves figuring out connectivity of joints to ends or edges.
	edge and node max intensity scores and edge lengths are stored as attributes. '''
def features_to_graph(joints, ends, edges):
	# print("    getting contours...")
	img1 = np.zeros_like(edges)
	img2 = np.zeros_like(edges)


	#give CV_CHAIN_APPROX_TC89_L1,CV_CHAIN_APPROX_TC89_KCOS a go?
	end_cnts, hierarchy = cv2.findContours(ends,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	j_cnts, hierarchy = cv2.findContours(joints,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	edg_cnts, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


	node_max_scores = []
	pixel_map = {}
	for j_id, j_cnt in enumerate(j_cnts):
		max_score = 0

		for p in j_cnt:
			pixel_map[(p[0][1], p[0][0])] = j_id
			max_score = max(max_score, joints[p[0][1], p[0][0]])

		node_max_scores.append(max_score)


	for end_id, end_cnt in enumerate(end_cnts):
		max_score = 0

		for p in end_cnt:
			pixel_map[(p[0][1], p[0][0])] = end_id + len(j_cnts)
			max_score = max(max_score, ends[p[0][1], p[0][0]])

		node_max_scores.append(max_score)


	# print("    identifying edge ends...")

	H, W = joints.shape
	c_connections = []
	c_max_scores = []
	for i, contour in enumerate(edg_cnts):

		connections = []
		max_score = 0
		connection_types = []
		for p in contour:
			p = np.array((p[0][1], p[0][0]))
			max_score = max(max_score, edges[p[0], p[1]])

			for xy_delta in xy_deltas:
				p2 = (p + xy_delta)

				if p2[0] == -1 or p2[0] == H:
					continue
				if p2[1] == -1 or p2[1] == W:
					continue

				p2 = tuple(p2)

				found = False

				if p2 in pixel_map and p2 not in connections:
					connections.append(pixel_map[p2])

		if len(connections) != 2:
			# edge case which should be ignored
			# e.g. single pixel attached to two of a joint's pixels
			connections = None

		c_connections.append(connections)
		c_max_scores.append(max_score)

	# print("    contructing graph object...")

	G = nx.MultiGraph()

	for idx, cnt in enumerate(j_cnts):
		G.add_node(idx, **{"type": "joint", "contour": cnt, "score" : node_max_scores[idx]})
	for idx, cnt in enumerate(end_cnts):
		G.add_node(idx + len(j_cnts), **{"type": "end", "contour": cnt, "score" : node_max_scores[idx + len(j_cnts)]})

	for idx, cnt in enumerate(edg_cnts):
		conns = c_connections[idx]
		if conns is None:
			continue

		G.add_edge(conns[0], conns[1], **{
			"contour": cnt, 
			"length": cv2.arcLength(cnt,True),
			"score" : c_max_scores[idx]
			})


	return G


''' set graph nodes/edges' scores to the maximum of the scores
	of their connected nodes/edges. This is a kind of 'diffusion' process. '''
def spread_scores(G, iterations=1):

	for it in range(iterations):
		for i, u in enumerate(G.nodes):
			max_score = 0
			for v in G[u]:
				for w in G[u][v]:
					max_score = max(G[u][v][w]["score"], max_score)

			G.nodes[u]["score"] = max_score

		for u, v, w in G.edges:
			G[u][v][w]["score"] = max(G.nodes[u]["score"], G.nodes[v]["score"])

	return G


''' remove graph edges which do not meet certain requirements.
	spurs are edges with a 'joint' on one side and an 'end' on the other.
	free segments are edges with an 'end' on both sides.
	- an edge shorter than min_length requires at least keep_score to not be removed.
	- an edge with a score below remove_score is removed. '''
def prune_graph(G, min_spur_length, keep_spur_score, remove_spur_score, min_free_segment_length, keep_free_segment_score, remove_free_segment_score):


	to_remove = []

	for u, v, w in G.edges:
		types = (G.nodes[u]["type"], G.nodes[v]["type"])
		if "joint" in types and "end" in types:
			if G[u][v][w]["length"] < min_spur_length and G[u][v][w]["score"] < keep_spur_score:
				to_remove.append((u, v, w))
			elif G[u][v][w]["score"] < remove_spur_score:
				to_remove.append((u, v, w))


	for u, v, w in G.edges:
		types = (G.nodes[u]["type"], G.nodes[v]["type"])
		if "joint" not in types:
			if G[u][v][w]["length"] < min_free_segment_length and G[u][v][w]["score"] < keep_free_segment_score:
				to_remove.append((u, v, w))
			elif G[u][v][w]["score"] < remove_free_segment_score:
				to_remove.append((u, v, w))

	G.remove_edges_from(to_remove)
	G = remove_unconnected(G)

	return G


''' remove edges which don't satisfy the condition
	imposed by filter_fn. '''
def filter_graph_edges(G, filter_fn):
	to_remove = []
	for u, v, w in G.edges:
		if not filter_fn(G[u][v][w]):
			to_remove.append((u, v, w))
	G.remove_edges_from(to_remove)
	G = remove_unconnected(G)
	return G



''' remove unconnected nodes '''
def remove_unconnected(G):
	to_remove = []
	for u in G.nodes:
		if nx.degree(G,u) == 0:
			to_remove.append(u)
	G.remove_nodes_from(to_remove)
	return G


''' order the pixels of a contour along a vessel 
	- this is achieved by starting at first_pt and
	  repeatedly finding the closest pixel. '''
def ordered_cnt_pixels(contour, first_pt):
	cnt = np.unique(contour, axis=0)
	curr_idx = np.argmin([np.sqrt(np.sum(np.square((first_pt[0] - p2[0])))) for p2 in cnt])

	if len(cnt) == 1:
		return cnt

	new_cnt = []
	while len(cnt) > 1:
		curr_pt = cnt[curr_idx]
		new_cnt.append(curr_pt)
		cnt = np.delete(cnt, curr_idx, axis=0)
		curr_idx = np.argmin([np.sqrt(np.sum(np.square((curr_pt[0] - p2[0])))) for p2 in cnt])

	return np.array(new_cnt)


''' order the pixels of all contours in the graph '''
def order_graph_cnt_pixels(G):
	for u, v, w in G.edges:
		cnt = G[u][v][w]["contour"]
		end = np.mean(G.nodes[u]["contour"], axis=0)
		G[u][v][w]["contour"] = ordered_cnt_pixels(cnt, end)
	
	return G


''' approximate each vessel as a chain of straight lines
	and store for each edge as a "point chain" attribute. '''
def make_graph_line_chains(G, max_deviation=2):
	for u, v, w in G.edges:
		cnt = cv2.approxPolyDP(G[u][v][w]["contour"], max_deviation, closed=False)
		G[u][v][w]["point chain"] = np.array([x[0] for x in cnt]).astype(np.int32)

	return G



''' construct and prune the skeleton for several iterations. '''
def simplify_graph(G, shape, params=None):

	if params is None:
		# default params (can be played with)
		params = [(1, 1, 2),
				 ((20.,     80,   0, 20., 50,  0), 
				  (20.,    120,   5, 20., 50,  5), 
				  (20., np.inf, 150, 20., 50, 50)),
				  (False, True, True)]

	# show_img(skel, "plasma")
	for i in range(len(params[0])):
		print("iteration", i+1)

		G = spread_scores(G, iterations=params[0][i])
		G = prune_graph(G, *params[1][i])

		# re-draw simplified skeleton and re-construct graph
		skel = graph_to_skel(G, shape, simplify=params[2][i])
		G = skel_to_graph(skel)

	# skel[skel > 0] = 1

	return G






# after investigation, cv2.approxPolyDP is identical but
# faster and issues with it were due to the structure of
# the contour data (non-uniqueness, non-orderedness)
# (kept just in case it still turns out to have issues)
#-----------------------------------------------------------------
# this is not a very good way to get straight line approximations
# because it is implemented in python so is slow.
# only implemented because it was quick and easy
# in order to get a functional tool.


''' perpendicular distance of pt from the line connecting start and end. '''
def perp_dist(start, end, pt):

	d1 = end - start
	d1_unit = d1 / np.hypot(*d1)
	d2 = pt - start

	cross_prod = np.cross(d1_unit, d2)

	return np.abs(cross_prod)


''' approximate contours with the Ramer-Douglas-Peucker algorithm.
	implemented manually in Python. '''
def approx_line_RDP(contour, max_d_allowed=1):
	if len(contour) <= 3:
		return contour

	start = contour[0]
	end = contour[-1]
	# print(contour)
	dists = [perp_dist(start[0], end[0], pt[0]) for pt in contour[1:-1]]
	# print(dists)
	d_max = np.max(dists)
	i = np.argmax(dists) + 1

	if d_max > max_d_allowed:
		# print("not allowed", contour[:i], contour[i:])
		cnt1 = approx_line_RDP(contour[:i],  max_d_allowed)
		cnt2 = approx_line_RDP(contour[i:],  max_d_allowed)
		results = np.concatenate((cnt1, cnt2))
	else:
		results = np.array([start, end])

	return results
# -----------------------------------------------------------------
