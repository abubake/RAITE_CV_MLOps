import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
         '''
         when registering an object we use the next available object
		ID to store the centroid
         '''
         self.objects[self.nextObjectID] = centroid
         self.disappeared[self.nextObjectID] = 0
         self.nextObjectID += 1

    def deregister(self, objectID):
        '''
        to deregister an object ID we delete the object ID from
		both of our respective dictionaries
        '''
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, boxes=None, centroids=None):
        '''
        Updates ID either given centroids or boxes
        '''
        if centroids: # calling boxes centroids for algorithm logic
            boxes = centroids
            
        if len(boxes) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                # Deregister if the object has disappeared too long
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects
        
        # When we do have boxes/ measurements:
        if centroids:
            inputCentroids = centroids # should check if this works
        else:  
            inputCentroids = np.zeros((len(boxes), 2), dtype="int")
            for (i, (startX, startY, endX, endY)) in enumerate(boxes):
                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)
                inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:  # Register our centroids if we don't have any yet
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)  # Compute distance between existing and input centroids

            rows = D.min(axis=1).argsort()  # Sort the minimum distances
            cols = D.argmin(axis=1)[rows]  # Get corresponding column indices

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue  # Skip if we've already seen this row or column

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]  # Update the centroid position
                self.disappeared[objectID] = 0  # Reset disappearance counter

                usedRows.add(row)
                usedCols.add(col)

            # Now we examine what we have not used
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # Handle unused rows (existing objects that didn't match with new centroids)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1  # Increment disappearance counter

                # Deregister if it has disappeared too long
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Handle unused columns (new objects that weren't matched)
            for col in unusedCols:
                self.register(inputCentroids[col])  # Register new centroids

        return self.objects

