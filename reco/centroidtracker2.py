
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import os
class CentroidTracker:
	def __init__(self, maxDisappeared=float(os.getenv('TRACKER_MAX_FRAME_LOSS')), maxDistance=float(os.getenv('TRACKER_MAX_DISTANCE'))):
		
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		
		self.maxDisappeared = maxDisappeared

		
		self.maxDistance = maxDistance

	def register(self, centroid):
		
		self.objects[self.nextObjectID] = {"centroid":centroid,"etudiants":{centroid[2]:1}}
		self.disappeared[self.nextObjectID] = 0
		if self.nextObjectID == 100:
			self.nextObjectID = 0
		else:
			self.nextObjectID += 1

	def deregister(self, objectID):
		
		etudiants = self.objects[objectID]["etudiants"]
		max_matricule = max(etudiants,key=etudiants.get)
		if etudiants[max_matricule] > 1:
			print("max",max_matricule, "id",objectID, "count",etudiants[max_matricule])
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		
		# is empty
		if len(rects) == 0:
			
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids1 = []
		

		
		for (i, (startX, startY, endX, endY,name)) in enumerate(rects):
			
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids1.append([cX,cY,name,startX, startY, endX, endY])

		inputCentroids1=np.array(inputCentroids1)
		#print(inputCentroids1[:,:2])

		
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids1)):
				self.register(inputCentroids1[i])

		else:
			
			objectIDs = list(self.objects.keys())
			objectCentroids = [obj["centroid"] for obj in self.objects.values()]
			objectCentroids=np.array(objectCentroids)
			#print(objectCentroids)

			D = dist.cdist(np.array(objectCentroids[:,:2].astype('int32')), inputCentroids1[:,:2].astype('int32'))
			# if D.size>1:
			# 	print(">1")
			
			rows = D.min(axis=1).argsort()

			cols = D.argmin(axis=1)[rows]

			
			usedRows = set()
			usedCols = set()

			
			for (row, col) in zip(rows, cols):
				
				if row in usedRows or col in usedCols:
					continue

				if D[row, col] > self.maxDistance:
					continue

				objectID = objectIDs[row]
				self.objects[objectID]["centroid"] = inputCentroids1[col]
				self.objects[objectID]["etudiants"][inputCentroids1[col][2]] = self.objects[objectID]["etudiants"].get(inputCentroids1[col][2],0)+1
				self.disappeared[objectID] = 0


				usedRows.add(row)
				usedCols.add(col)


			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			if D.shape[0] >= D.shape[1]:

				for row in unusedRows:

					objectID = objectIDs[row]
					self.disappeared[objectID] += 1


					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			else:
				for col in unusedCols:
					self.register(inputCentroids1[col])

		#print(self.objects)

		return self.objects
