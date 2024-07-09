import numpy as np
from itertools import product


inf = float('inf')


def findPatchHorizontal(refBlock, texture, blocksize, overlap, tolerance, rng: np.random.Generator):
	'''
	Find best horizontal match from the texture
	'''
	H, W = texture.shape[:2]
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf
	for i, j in product(range(H-blocksize), range(W-blocksize)):
		rmsVal = ((texture[i:i+blocksize, j:j+overlap] - refBlock[:, -overlap:])**2).mean()
		if rmsVal > 0:
			errMat[i, j] = rmsVal

	minVal = np.min(errMat)
	y, x = np.nonzero(errMat < (1.0 + tolerance)*(minVal))
	c = rng.integers(len(y))
	y, x = y[c], x[c]
	return texture[y:y+blocksize, x:x+blocksize]

def findPatchBoth(refBlockLeft, refBlockTop, texture, blocksize, overlap, tolerance, rng: np.random.Generator):
	'''
	Find best horizontal and vertical match from the texture
	'''
	H, W = texture.shape[:2]
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf
	for i, j in product(range(H-blocksize), range(W-blocksize)):
		rmsVal = ((texture[i:i+overlap, j:j+blocksize] - refBlockTop[-overlap:, :])**2).mean()
		rmsVal = rmsVal + ((texture[i:i+blocksize, j:j+overlap] - refBlockLeft[:, -overlap:])**2).mean()
		if rmsVal > 0:
			errMat[i, j] = rmsVal

	minVal = np.min(errMat)
	y, x = np.nonzero(errMat < (1.0 + tolerance)*(minVal))
	c = rng.integers(len(y))
	y, x = y[c], x[c]
	return texture[y:y+blocksize, x:x+blocksize]


def findPatchVertical(refBlock, texture, blocksize, overlap, tolerance, rng: np.random.Generator):
	'''
	Find best vertical match from the texture
	'''
	H, W = texture.shape[:2]
	errMat = np.zeros((H-blocksize, W-blocksize)) + inf
	for i, j in product(range(H-blocksize), range(W-blocksize)):
		rmsVal = ((texture[i:i+overlap, j:j+blocksize] - refBlock[-overlap:, :])**2).mean()
		if rmsVal > 0:
			errMat[i, j] = rmsVal

	minVal = np.min(errMat)
	y, x = np.nonzero(errMat < (1.0 + tolerance)*(minVal))
	c = rng.integers(len(y))
	y, x = y[c], x[c]
	return texture[y:y+blocksize, x:x+blocksize]


def getMinCutPatchHorizontal(block1, block2, blocksize, overlap):
	'''
	Get the min cut patch done horizontally
	'''
	err = ((block1[:, -overlap:] - block2[:, :overlap])**2).mean(2)
	# maintain minIndex for 2nd row onwards and
	minIndex = []
	E = [list(err[0])]
	for i in range(1, err.shape[0]):
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]
		e = np.array([e[:-2], e[1:-1], e[2:]])
		# Get minIndex
		minArr = e.min(0)
		minArg = e.argmin(0) - 1
		minIndex.append(minArg)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr
		E.append(list(Eij))

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask = np.zeros((blocksize, blocksize, block1.shape[2]))
	for i in range(len(path)):
		mask[i, :path[i]+1] = 1

	resBlock = np.zeros(block1.shape)
	resBlock[:, :overlap] = block1[:, -overlap:]
	resBlock = resBlock*mask + block2*(1-mask)
	# resBlock = block1*mask + block2*(1-mask)
	return resBlock


def getMinCutPatchVertical(block1, block2, blocksize, overlap):
	'''
	Get the min cut patch done vertically
	'''
	resBlock = getMinCutPatchHorizontal(np.rot90(block1), np.rot90(block2), blocksize, overlap)
	return np.rot90(resBlock, 3)


def getMinCutPatchBoth(refBlockLeft, refBlockTop, patchBlock, blocksize, overlap):
	'''
	Find minCut for both and calculate
	'''
	err = ((refBlockLeft[:, -overlap:] - patchBlock[:, :overlap])**2).mean(2)
	# maintain minIndex for 2nd row onwards and
	minIndex = []
	E = [list(err[0])]
	for i in range(1, err.shape[0]):
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]
		e = np.array([e[:-2], e[1:-1], e[2:]])
		# Get minIndex
		minArr = e.min(0)
		minArg = e.argmin(0) - 1
		minIndex.append(minArg)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr
		E.append(list(Eij))

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask1 = np.zeros((blocksize, blocksize, patchBlock.shape[2]))
	for i in range(len(path)):
		mask1[i, :path[i]+1] = 1

	###################################################################
	## Now for vertical one
	err = ((np.rot90(refBlockTop)[:, -overlap:] - np.rot90(patchBlock)[:, :overlap])**2).mean(2)
	# maintain minIndex for 2nd row onwards and
	minIndex = []
	E = [list(err[0])]
	for i in range(1, err.shape[0]):
		# Get min values and args, -1 = left, 0 = middle, 1 = right
		e = [inf] + E[-1] + [inf]
		e = np.array([e[:-2], e[1:-1], e[2:]])
		# Get minIndex
		minArr = e.min(0)
		minArg = e.argmin(0) - 1
		minIndex.append(minArg)
		# Set Eij = e_ij + min_
		Eij = err[i] + minArr
		E.append(list(Eij))

	# Check the last element and backtrack to find path
	path = []
	minArg = np.argmin(E[-1])
	path.append(minArg)

	# Backtrack to min path
	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	# Reverse to find full path
	path = path[::-1]
	mask2 = np.zeros((blocksize, blocksize, patchBlock.shape[2]))
	for i in range(len(path)):
		mask2[i, :path[i]+1] = 1
	mask2 = np.rot90(mask2, 3)


	mask2[:overlap, :overlap] = np.maximum(mask2[:overlap, :overlap] - mask1[:overlap, :overlap], 0)

	# Put first mask
	resBlock = np.zeros(patchBlock.shape)
	resBlock[:, :overlap] = mask1[:, :overlap]*refBlockLeft[:, -overlap:]
	resBlock[:overlap, :] = resBlock[:overlap, :] + mask2[:overlap, :]*refBlockTop[-overlap:, :]
	resBlock = resBlock + (1-np.maximum(mask1, mask2))*patchBlock
	return resBlock


# generateTextureMap -> was moved to quilting.py; renamed to generate_texture; added data/behavior for comfyui node
