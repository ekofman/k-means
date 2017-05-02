import sys, re, random, math
from sets import Set
import argparse
import numpy as np

# Usage:python kmeans.py k data_file max_iterations [centroid_file] 

"""
This DataPoint class defines an entity that contains an id number and a
vector of features that belong to the datapoint.
"""
class DataPoint:
    def __init__(self, number, vector):
        self.vector = vector
        self.number = number
    
    def __str__(self):
        string = str(self.number) + ": "  + str(self.vector)
        return string

"""
Given a filename of a file containing the coordinates of centroids, this
function reads centroids into an array of tuples, such that each tuple
represents the coordinates of one centroid.
"""
def read_centroids(fileName):
    result_array = []
    f = open(fileName, 'r')
    lines = f.readlines()

    for line in lines: 
        line = re.split("\t", line)
        row_array = []
        for i in range(len(line)):
            row_array.insert(0,float(line[i-1]))
        result_array.append(row_array)
    return result_array

"""
Given the name of a file containing datapoint tuples, creates and returns an array of
DataPoint objects
"""
def read_test_data(file_name):
    data_point_array = []
    f = open(file_name, 'r')
    lines = f.readlines()

    for j, line in enumerate(lines): 
        line = re.split("\t", line)
        row_array = []
        for i in range(len(line)):
            row_array.insert(0,float(line[i-1]))
        # First index of each patient array is the patient number
        patient = DataPoint(j, row_array)
        data_point_array.append(patient)
    return data_point_array

"""
Given "k", the number of centroids (and ultimately, clusters) desired, and
the array of DataPoint objects read in originally, randomly chooses "k" datapoint
coordinates to be the starting centroids
"""
def generate_centroids_randomly(k, test_data):
    rand_indices_set = set()
    while len(rand_indices_set) < k:
        rand_index = random.randint(0, len(test_data)-1)
        rand_indices_set.add(rand_index)
    rand_centroids = []
    for index in rand_indices_set:
        rand_centroids.append(test_data[index].vector)
    return rand_centroids

"""
Given two vectors, calculates the euclidian distance including all
attributes
"""
def get_euclidian_distance(vector_1, vector_2):
    total_distance = 0
    for i in xrange(len(vector_1)):
        attribute_distance = (vector_1[i] - vector_2[i])**2
        total_distance = total_distance + attribute_distance
    euclidian_distance = math.sqrt(total_distance)
    return euclidian_distance

def get_centroid(cluster):
    """Return the centroid for a cluster of points"""
    num_points = float(len(cluster))
    num_attributes = len(cluster[0].vector) 
    attribute_sums = np.zeros(shape=(num_attributes, 1))
    for point in cluster:
        for j, attribute in enumerate(point.vector):
            attribute_sums[j] = attribute_sums[j] + attribute
    new_centroid = []
    for i in xrange(num_attributes):
        new_centroid.append(round(float(attribute_sums[i])/num_points, 3))
    return new_centroid

"""
Given the current set of clusters and the current set of centroids,
calculates whether any patients should be changed to a different cluster,
then recalculates the centroids for the new clusters. Returns both the
new clusters and the new centroids.
"""
def generate_new_clusters(clusters, centroids):
    new_clusters = []
    new_centroids = []

    # Initalize new cluster array
    for n in xrange(len(centroids)):
        new_clusters.append([])

    # For each point, check distance from all centroids and assign point to
    # cluster of closest centroid using Euclidian distance
    for cluster in clusters:
        for point in cluster:
            min_distance = sys.maxint
            cluster_assignment = None
            for i, centroid in enumerate(centroids):
                distance = get_euclidian_distance(point.vector, centroid)
                if distance < min_distance:
                    min_distance = distance
                    # Point has chosen a cluster
                    cluster_assignment = i 
            new_clusters[cluster_assignment].append(point)

    # Calculate centroids of each new cluster
    for cluster in new_clusters:
        new_centroid = get_centroid(cluster)
        new_centroids.append(new_centroid)
    return([new_clusters, new_centroids])

def centroid_sum_distance(cluster):
    """Simply sum the values for each element of the cluster's vector"""
    total_sum = 0
    for point in cluster:
        total_sum = total_sum + sum(point.vector)
    return total_sum

"""
main function, runs entire program
"""    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run k-means clustering algorithm')
    parser.add_argument('k', metavar='k', type=int)
    parser.add_argument('max_iterations', metavar='max_iterations', type=int)
    parser.add_argument('data_path', metavar='data_path', type=str)
    parser.add_argument('--centroids_path', metavar='centroids_path', type=str)

    args = parser.parse_args()
    k = args.k
    max_iterations = args.max_iterations
    data_path = args.data_path
    
    testData = read_test_data(data_path)    
    if k > len(testData):
        sys.stdout.write("More clusters than data points specified\n")
    if len(sys.argv) < 5:
        sys.stdout.write("No centroids specified ... generating randomly\n")
        centroids = generate_centroids_randomly(k, testData)
    else:
        sys.stdout.write("Centroids specified\n")
        centroids = read_centroids(centroids_path)
    
    # Starting config is simply one blob cluster with all points
    clusters = [testData] 
    
    # Run iterations
    [new_clusters, new_centroids] =  generate_new_clusters(clusters, centroids)
    iteration = 0
    while new_clusters != clusters and new_centroids != centroids and iteration <= max_iterations:
        iteration = iteration + 1           
        sys.stdout.write("Running another cluster generation... {}\n".format(iteration + 1))
        clusters = new_clusters
        centroids = new_centroids
        [new_clusters, new_centroids] = generate_new_clusters(clusters, centroids)
    sys.stdout.write("Converged. Final clusters: \n")

    # Print the output
    for i, new_cluster in enumerate(sorted(new_clusters, key=centroid_sum_distance)):
        sys.stdout.write("Cluster {}:\n".format(i))
        for point in new_cluster:
            sys.stdout.write("\t{}\n".format(point))
