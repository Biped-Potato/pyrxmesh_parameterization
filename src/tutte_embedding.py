import igl
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp

def tutte_embedding(mesh):
    V = mesh.vertices()
    F = mesh.faces()

    #loop indices
    b = igl.boundary_loop(F)
    #loop vertices(as a circle)
    bc = igl.map_vertices_to_circle(V, b)


    n = int(mesh.num_vertices)

    #adjacency matrix 
    adj = igl.adjacency_matrix(F).astype(float).tocsr()

    #degrees along the diagonal
    deg = np.zeros(n)
    for i in range(n):
        deg[i] = adj.indptr[i + 1] - adj.indptr[i]

    uv = np.zeros((mesh.num_vertices,2))
    for i in range(len(b)):
        uv[b[i]] = [bc[i][0],bc[i][1]]
        adj.data[adj.indptr[b[i]]:adj.indptr[b[i] + 1]] = 0
        deg[b[i]] = -1

    #solve matrix
    a = (adj - sp.diags(deg)).tocsc()
    lu = spla.splu(a)
    return lu.solve(uv)