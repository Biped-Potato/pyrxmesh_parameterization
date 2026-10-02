from pathlib import Path
from src.tutte_embedding import tutte_embedding

import argparse
import pyrxmesh as rx
import pyrxmesh_parameterization
import igl
import torch
import polyscope as ps
import numpy as np
import time

def output_uvs(mesh, uv, args):
    with open(str(Path(__file__).parent / "output_uvs" / args.obj_file_name), "w") as f:
        indexing = mesh.linear_to_global(element_kind=rx.ElementKind.Vertex)
        uv_global = np.empty_like(uv)
        
        for i in range(len(uv_global)):
            uv_global[indexing[i]] = uv[i]
        
        for i in range(uv_global.shape[0]):
            f.write(f"v {uv_global[i][0]} {uv_global[i][1]} {0.0} \n",)

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj-file-name",   default="bunnyhead.obj", type=str)
    parser.add_argument("--output-folder",   default=Path(__file__), type=Path)
    parser.add_argument("--uv-file-name",    default="", type=str)
    parser.add_argument("--device-id",       default=0, type=int)
    parser.add_argument("--learning-rate",   default=1e-9, type=float)
    parser.add_argument("--num-iter", default=100, type=int)
    parser.add_argument("--polyscope", default=1,type=int)

    args = parser.parse_args()

    torch.cuda.set_device(args.device_id)
    
    rx.init(args.device_id)
    mesh = rx.RXMeshStatic(str(Path(__file__).parent / "meshes" / args.obj_file_name))

    if mesh.is_closed():
        raise ValueError("The input mesh is closed. The input mesh should have boundaries.")

    if args.polyscope == 1:
        ps.init()
        ps_mesh = ps.register_surface_mesh("mesh", mesh.vertices(), mesh.faces())

    coordinates = mesh.input_vertex_coordinates()
    
    #use a column major vector of dimension 4 instead of a 2x2 matrix
    rest_shape = mesh.add_face_attribute("fRestShape", dtype="float32", dim=4)

    #initialize uv coordinates attribute in mesh
    uv_attr = mesh.add_vertex_attribute("uv", dtype="float32", dim=2)

    if not args.uv_file_name:
        uv = tutte_embedding(mesh)
        print("generated new uvs")
        uv_attr.from_numpy_copy(uv, target="all")
        output_uvs(mesh,uv,args)
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
    
    if args.polyscope == 1:
        ps_mesh.add_parameterization_quantity("input_uv", uv_attr.to_numpy_copy(source="device"))

    #perform gradient descent
    gradient = torch.empty(uv.shape, dtype=uv.dtype, device=uv.device)

    optimizer = torch.optim.SGD(
        [uv],
        lr = args.learning_rate,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = energy.value_and_grad(uv, out=gradient, copy = "never")
        uv.grad = gradient
        return loss

    starting_energy = energy.value_and_grad(uv, out=gradient, copy = "never")

    start_time = time.perf_counter()
    
    for i in range(args.num_iter):
        loss = optimizer.step(closure)
    else:
        ending_energy = loss.detach().cpu().numpy()

    end_time = time.perf_counter()

    print(f"iterations = {args.num_iter}, energy = {starting_energy:.6f} -> {ending_energy:.6f}, time = {(end_time - start_time) * 1000:.6f} ms")

    rx.cuda_stream_synchronize()
    if args.polyscope == 1:
        ps_mesh.add_parameterization_quantity("uv", uv_attr.to_numpy_copy(source="device"))
        ps.show()
    

if __name__ == "__main__":
    main()