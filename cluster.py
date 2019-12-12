import pandas as pd
import math
import json
import numpy as np
from functools import lru_cache
from multiprocessing import Pool

pd.options.mode.chained_assignment = None

class Point:
    def __init__(self, pattern_id):
        self.length = len(pattern_id)
        self.pattern_id = pattern_id
        self.z = -1 # number of the point in the cluster, with -1 being centroid

    def __str__(self):
        return str(self.pattern_id)

    def toJSON(self):
        return {
            'pattern_id':self.pattern_id
        }


class Cluster:
    def __init__(self, dim, centroid):
        self.dim = dim
        # center of the cluster
        self.centroid = centroid
        # points belonging to the cluster
        self.points = []
        # list of the distances between centroid and points in the same order
        self.distances = []

    # this method finds the average distance of all elements in cluster to its centroid
    def computeS(self):
        n = len(self.points)
        if n == 0:
            return 0
        s = 0
        for x in self.distances:
            s += x
        return float(s / n)


class Clustering:
    def __init__(self, generation, data, kmax):
        self.generation = generation
        self.data = data
        self.dim = data.shape[1]
        self.penalty = 1000000
        self.kmax = kmax

    # Uses de Davies Bouldin method to return the average of the R metric, as a
    # metric of how good the clusters are
    def daviesBouldin(self, clusters):
        sigmaR = 0.0
        nc = len(clusters)
        # For each cluster
        for i in range(nc):
            # Accumulate the R of all clusters
            sigmaR = sigmaR + self.computeR(clusters)
            #print(sigmaR)
        # Given the average the R of all clusters
        DBIndex = float(sigmaR) / float(nc)
        return DBIndex

    def computeR(self, clusters):
        listR = []
        # For each cluster pair (that it's not with itself), append to a list of
        # the sum of the average distance of the clusters centroid to their
        # points, divided by the distance of the two centroids
        for i, iCluster in enumerate(clusters):
            for j, jCluster in enumerate(clusters):
                if(i != j):
                    temp = self.computeRij(iCluster, jCluster)
                    listR.append(temp)
        # Returns the metric for the worst cluster
        return max(listR)

    # Given two clusters, computes the sum of the average distance of their
    # centroid to it's points and divide it by the euclidianDistance of the two
    # centroids
    def computeRij(self, iCluster, jCluster):
        Rij = 0

        d = self.euclidianDistance(
            iCluster.centroid, jCluster.centroid)
        #print("d",d)
        #print("icluster",iCluster.computeS())
        Rij = (iCluster.computeS() + jCluster.computeS()) / d

        #print("Rij:", Rij)
        return Rij

    @lru_cache(maxsize=16_384)
    def euclidianDistance(self, point1, point2):
        sum = 0
        for i in range(0, point1.length):
            square = pow(
                point1.pattern_id[i] - point2.pattern_id[i], 2)
            sum += square

        sqr = math.sqrt(sum)
        return sqr

    # Given all clusters, appends all data points to the closest cluster and
    # returns the groups
    def calcDistance(self, clusters):
        kmax = self.kmax
        dim = self.dim
        data = self.data
        dis = 0
        disSet = []

        # For each point in the data sample
        for z in range(data.shape[0]):
            # Creates this sample point
            point = Point(data.loc[z][0:dim])
            # Puts z with the number of the individual
            point.z = z

            # For each cluster
            for i in range(kmax):
                # Calculate the distance of this point to each cluster and
                # append the distance to the distance set
                dis = self.euclidianDistance(clusters[i].centroid, point)
                disSet.append(dis)
                dis = 0

            # Given the array with this point distance to each cluster,
            # checks the cluster with the minimum distance and append the point
            # to this cluster
            clusters = self.findMin(
                disSet, clusters, point)
            # clear distSet
            disSet = []

        # Returns the cluster list with appended points to each one of them
        return clusters

    # Given a set of clusters and a point, with their respective distances, find
    # the pair with minimun distance and appends the point to the cluster
    def findMin(self, disSet, clusters, point):
        # Gets the cluster with smallest distance
        n = disSet.index(min(disSet))  # n is index
        minDis = disSet[n]
        # Appends the point to the cluster
        clusters[n].points.append(point)
        # Append the point distance to the cluster
        clusters[n].distances.append(minDis)

        # Return the clusters with the appended point
        return clusters

    # childChromosome, kmax
    def calcChildFit(self, childChromosome):
        kmax = self.kmax
        dim = self.dim
        clusters = []
        for j in range(kmax):
            point = Point(childChromosome.genes[j * dim: (j + 1) * dim])
            clusters.append(Cluster(dim, point))

        clusters = self.calcDistance(clusters)
        DBIndex = self.daviesBouldin(clusters)

        childChromosome.fitness = 1 / DBIndex

        return childChromosome

    def calculateChromossomeFitness(self, i):
        dim = self.dim
        clusters = []

        # For each group/cluster
        for j in range(self.kmax):
            # Creates point with all the features of this group at this
            # individual
            point = Point(
                self.generation.chromosomes[i].genes[j * dim: (j + 1) * dim]
            )
            clusters.append(Cluster(dim, point))

        # Add the data points to the cluster with the closest centroid
        clusters = self.calcDistance(clusters)
        # Evaluate how good the clusters are getting the DBI index
        DBIndex = self.daviesBouldin(clusters)
        # Generates the population fitness by dividing 1 by the DBI
        return (1 / DBIndex)

    # Classifies the data points between the clusters and calculates their
    # fitness using daviesBouldin
    def calcChromosomesFit(self):
        print('\t\tCalculating fitness...')
        kmax = self.kmax
        generation = self.generation
        numOfInd = generation.numberOfIndividual
        data = self.data
        chromo = generation.chromosomes

        # For every indivudal/chromossome
        args = list(range(0, numOfInd))
        pool = Pool()
        fitness = pool.map(self.calculateChromossomeFitness, args)
        for i in range(0, numOfInd):
            generation.chromosomes[i].fitness = fitness[i]

        return generation

    # Print for the best chromosome in the generation
    def printIBest(self, iBest):
        kmax = self.kmax
        dim = self.dim
        clusters = []

        # Extract each cluster centroid from the chromosome
        for j in range(kmax):
            point = Point(iBest.genes[j * dim: (j + 1) * dim])
            clusters.append(Cluster(dim, point))

        # Calc distance and bouldin
        clusters = self.calcDistance(clusters)
        # DBIndex = self.daviesBouldin(clusters)
        # z = (np.zeros(self.data.shape[0])).tolist()
        # for i, cluster in enumerate(clusters):
        #     for j in cluster.points:
        #         z[j.z] = i

        # correct_answer = 0
        # cluster_size = math.floor(self.data.shape[0]/self.kmax)
        # # For each cluster
        # for k in range(0, self.kmax):
        #     # For each
        #     for i in range(0, cluster_size):
        #         if z[i] == k:
        #             correct_answer += 1

        # accuracy = (correct_answer / self.data.shape[0]) * 100

        # print("accuracy :", accuracy)
        print("iBest Fitness:", 1 / DBIndex)
        # print("all index:", z)
        # print("Clusters centroid:")
        # for i, cluster in enumerate(clusters):
        #     print("centroid", i, " :", cluster.centroid)

    def output_result(self, iBest, data):
        print("Saving the result...")
        kmax = self.kmax
        dim = self.dim
        clusters = []
        for j in range(kmax):
            point = Point(iBest.genes[j * dim: (j + 1) * dim])
            clusters.append(Cluster(dim, point))

        clusters = self.calcDistance(clusters)
        centroids = []
        for i in range(kmax):
            centroids.append(clusters[i].centroid)
        z = (np.zeros(self.data.shape[0])).tolist()
        for i, cluster in enumerate(clusters):
            for j in cluster.points:
                z[j.z] = i

        with open('result/cluster_center.json', 'w') as outfile:
            json.dump([e.toJSON() for e in centroids], outfile, sort_keys=True,
                      indent=4, separators=(',', ': '))

        # rename df header
        col_name = list()
        for i in range(data.shape[1]):
            col_name.append("f{0}".format(i))
        data.columns = col_name

        # insert cluster result
        data['Cluster Index'] = pd.Series(z, index=data.index)
        data.to_csv('result/result.csv', index=None)
        print("Done.")