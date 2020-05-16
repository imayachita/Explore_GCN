import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#Initialize the graph
G = nx.Graph()

#Create nodes
#In this example, the graph will consist of 6 nodes.
#Each node is assigned node feature which corresponds to the node name
for i in range(6):
    G.add_node(i, name=i)


#Define the edges and the edges to the graph
edges = [(0,1),(0,2),(1,2),(0,3),(3,4),(3,5),(4,5)]
G.add_edges_from(edges)

#Inspect the node features
print('Graph Nodes: ', G.nodes.data())

#Plot the graph
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

#Get the Adjacency Matrix (A) and Node Features Matrix (X) as numpy array
A = np.array(nx.attr_matrix(G, node_attr='name')[0])
X = np.array(nx.attr_matrix(G, node_attr='name')[1])


#Dot product Adjacency Matrix (A) and Node Features (X)
AX = np.dot(A,X)
print("Dot product of A and X (AX): ", AX)

#Add Self Loops
G_self_loops = G.copy()

self_loops = []
for i in range(6):
    self_loops.append((i,i))

G_self_loops.add_edges_from(self_loops)


#Check the edges of G_self_loops after adding the self loops
print('Edges of G with self-loops: ', G_self_loops.edges)


#Get the Adjacency Matrix (A) and Node Features Matrix (X) of added self-lopps graph
A_hat = np.array(nx.attr_matrix(G_self_loops, node_attr='name')[0])
print('Adjacency Matrix of added self-loops G (A_hat): ', A_hat)

#Calculate the dot product of A_hat and X (AX)
A_hat_X = np.dot(A_hat, X)


#Get the Degree Matrix of the added self-loops graph
Deg_Mat = G_self_loops.degree()
print('Degree Matrix of added self-loops G (D): ', Deg_Mat)


#Convert the Degree Matrix to a N x N matrix where N is the number of nodes
d = [deg for (n,deg) in list(G_self_loops.degree())]
D = np.diag(d)
print('Degree Matrix of added self-loops G as numpy array (D): ', D)


#Find the inverse of Degree Matrix (D)
D_inv = np.linalg.inv(D)
print('Inverse of D: ', D_inv)


print('Shape of A_hat: ', A_hat.shape)
print('Shape of X: ', X.shape)
X = np.expand_dims(X,axis=1)


#Normalized AX
DAX = np.dot(D_inv,A_hat_X)
print('Normalized AX: ', DAX)

#Symmetrically-normalized AX
D_half_norm = fractional_matrix_power(D, -0.5)
DAXD = np.dot(np.dot(D_half_norm,A_hat_X),D_half_norm)
print('Symmetrically-normalized AX: ', DAXD)

#Initialize the weights
np.random.seed(77777)
W0 = np.random.randn(X.shape[1],4) * 0.01
W1 = np.random.randn(W0.shape[1],2) * 0.01

#Implement ReLu as activation function
def relu(Z):
    return np.maximum(0,Z)


#Build GCN layer
#In this function, we implement numpy to simplify
def gcn(A,H,W):
    I = np.identity(A.shape[0]) #create Identity Matrix of A
    A_hat = A + I #add self-loop to A
    D = np.diag(np.sum(A_hat, axis=0)) #create Degree Matrix of A
    D_half_norm = fractional_matrix_power(D, -0.5) #calculate D to the power of -0.5
    eq = D_half_norm.dot(A_hat).dot(D_half_norm).dot(H).dot(W)
    return relu(eq)


#Do forward propagation
H1 = gcn(A,X,W0)
H2 = gcn(A,H1,W1)
print('Features Representation from GCN output: ', H2)


#Plot the features representation
x = H2[:,0]
y = H2[:,1]

marker_size = 1000

plt.scatter(x,y,marker_size)
plt.xlim([np.min(x)*0.9, np.max(x)*1.1])
plt.ylim([-1, 1])
plt.xlabel('Feature Representation Dimension 0')
plt.ylabel('Feature Representation Dimension 1')
plt.title('Feature Representation')


for i,row in enumerate(H2):
    str = "{}".format(i)
    plt.annotate(str, (row[0],row[1]),fontsize=18, fontweight='bold')

plt.show()
