from pathlib import Path

import argparse
import pyrxmesh as rx
import pyrxmesh_parameterization
import igl
import torch
import polyscope as ps
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj-file-name",   default="bunnyhead.obj", type=str)
    #parser.add_argument("--output-folder",   default=Path(__file__), type=Path)
    parser.add_argument("--uv-file-name",    default="", type=str)
    parser.add_argument("--solver",          default="cudss_choi", type=str)
    parser.add_argument("--device-id",       default=0, type=int)
    parser.add_argument("--cg-abs-tol",      default=1e-6, type=float)
    parser.add_argument("--cg-rel-tol",      default=0.0, type=float)
    parser.add_argument("--cg-max-iter",     default=10, type=int)
    parser.add_argument("--max-iter", default=100, type=int)
    parser.add_argument("--lbfgs-max-iter", default=100, type=int)

    args = parser.parse_args()

    torch.cuda.set_device(args.device_id)
    
    rx.init(args.device_id)
    mesh = rx.RXMeshStatic(str(Path(__file__).parent / "meshes" / args.obj_file_name))

    if mesh.is_closed():
        raise ValueError("The input mesh is closed. The input mesh should have boundaries.")

    ps.init()
    ps_mesh = ps.register_surface_mesh("mesh", mesh.vertices(), mesh.faces())

    coordinates = mesh.input_vertex_coordinates()
    
    #use a column major vector of dimension 4 instead of a 2x2 matrix
    rest_shape = mesh.add_face_attribute("fRestShape", dtype="float32", dim=4)

    #initialize uv coordinates attribute in mesh
    uv_attr = mesh.add_vertex_attribute("uv", dtype="float32", dim=2)
    if not args.uv_file_name:
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
            adj[b[i]] = np.zeros(n)
            deg[b[i]] = -1

        #solve matrix
        a = adj - sp.diags(deg)
        lu = spla.splu(a)
        uv = lu.solve(uv)
    
        #uv = igl.harmonic(V, F, b, bc, 1)

        
        with open(str(Path(__file__).parent / "output_uvs" / args.obj_file_name), "w") as f:
            indexing = mesh.linear_to_global(element_kind=rx.ElementKind.Vertex)
            uv_global = np.empty_like(uv)
            
            for i in range(len(uv_global)):
                uv_global[indexing[i]] = uv[i]
            
            for i in range(uv_global.shape[0]):
                f.write(f"v {uv_global[i][0]} {uv_global[i][1]} {0.0} \n",)

        uv_attr.from_numpy_copy(uv, target="all")
    else:
        #uv here would be UV coordinates with z = anything and F are indices
        (uv, fv) = igl.read_triangle_mesh(str(Path(__file__).parent / "input_uvs" / args.uv_file_name))
        if uv.shape[0] != mesh.num_vertices:
            raise ValueError(
                f"Number of vertices in the input UV file {uv.shape[0]} does not match \
                the number of vertices in the mesh {mesh.num_vertices}."
            )
        indexing = mesh.linear_to_global(rx.ElementKind.Vertex)
        uv_attr.from_numpy_copy(uv[indexing,:2], target="all")

    

    pyrxmesh_parameterization.compute_rest_shape(mesh,coordinates,rest_shape)

    rx.cuda_stream_synchronize()

    energy = pyrxmesh_parameterization.make_energy(mesh, rest_shape)

    uv = uv_attr.to_torch("device").detach().requires_grad_(True)

    ps_mesh.add_parameterization_quantity("input_uv", uv_attr.to_numpy_copy(source="device"))

    #perform gradient descent
    gradient = torch.empty(uv.shape, dtype=uv.dtype, device=uv.device)

    # optimizer = torch.optim.LBFGS(
    #     [uv],
    #     lr=1e-7,
    #     max_iter = args.lbfgs_max_iter,
    #     history_size = 20,
    #     tolerance_grad = 1e-7,
    #     tolerance_change = 1e-9,
    #     #line_search_fn = "none",
    # )

    optimizer = torch.optim.SGD(
        [uv],
        lr = 1e-9,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        e = energy.value_and_grad(uv, out=gradient, copy = "never")
        uv.grad = gradient
        return e

    prev_loss = 0
    tolerance = 0.001
    
    for i in range(100):
        loss = optimizer.step(closure)
        loss_val = loss.detach().cpu().numpy()
        if abs(prev_loss - loss_val) <= tolerance:
            print("energy is optimized")
            break
        if i == 0:
            print(loss_val)
        prev_loss = loss_val
    print(loss_val)
    

    rx.cuda_stream_synchronize()
    
    values = uv_attr.to_numpy_copy(source="device")

    ps_mesh.add_parameterization_quantity("uv", uv_attr.to_numpy_copy(source="device"))

    ps.show()

if __name__ == "__main__":
    main()