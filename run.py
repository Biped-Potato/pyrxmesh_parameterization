from pathlib import Path

import argparse
import pyrxmesh as rx
import pyrxmesh_parameterization
import igl
import torch
import polyscope as ps

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--obj-file-name",   default=Path(__file__).parent / "meshes" / "bunnyhead.obj", type=Path)
    #parser.add_argument("--output-folder",   default=Path(__file__), type=Path)
    parser.add_argument("--uv-file-name",    default="", type=str)
    parser.add_argument("--solver",          default="cudss_choi", type=str)
    parser.add_argument("--device-id",       default=0, type=int)
    parser.add_argument("--cg-abs-tol",      default=1e-6, type=float)
    parser.add_argument("--cg-rel-tol",      default=0.0, type=float)
    parser.add_argument("--cg-max-iter",     default=10, type=int)
    parser.add_argument("--lbfgs-max-iter", default=100, type=int)

    args = parser.parse_args()

    torch.cuda.set_device(args.device_id)
    
    rx.init(args.device_id)
    mesh = rx.RXMeshStatic(str(args.obj_file_name))

    if mesh.is_closed():
        raise ValueError("The input mesh is closed. THe input mesh should have boundaries.")

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
        b = igl.boundary_loop(F)
        bc = igl.map_vertices_to_circle(V, b)
        #pass this as the optimization variable
        uv = igl.harmonic(V, F, b, bc, 1)
        uv_attr.from_numpy_copy(uv, target="all")
    else:
        #uv here would be UV coordinates with z = anything and F are indices
        (uv, fv) = igl.read_triangle_mesh(args.uv_file_name)
        if uv.shape[0] != mesh.num_vertices:
            raise ValueError(
                f"Number of vertices in the input UV file {uv.shape[0]} does not match \
                the number of vertices in the mesh {mesh.num_vertices}."
            )
        uv_attr.from_numpy_copy(uv[:,:2], target="all")

    

    pyrxmesh_parameterization.compute_rest_shape(mesh,coordinates,rest_shape)

    energy = pyrxmesh_parameterization.make_energy(mesh, rest_shape)

    uv = uv_attr.to_torch("device").detach().requires_grad_(True)

    ps_mesh.add_parameterization_quantity("input_uv", uv_attr.to_numpy_copy(source="device"))

    #perform gradient descent
    gradient = torch.empty(uv.shape, dtype=uv.dtype, device=uv.device)

    optimizer = torch.optim.LBFGS(
        [uv],
        lr=1.0,
        max_iter = args.lbfgs_max_iter,
        history_size = 20,
        tolerance_grad = 1e-7,
        tolerance_change = 1e-9,
        line_search_fn ="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = energy.value_and_grad(uv, out=gradient, copy = "never")
        uv.grad = gradient
        return loss

    loss = optimizer.step(closure)

    rx.cuda_stream_synchronize()
    
    values = uv_attr.to_numpy_copy(source="device")

    ps_mesh.add_parameterization_quantity("uv", uv_attr.to_numpy_copy(source="device"))

    ps.show()
    
    for i in range(len(values)):
        print(values[i])

if __name__ == "__main__":
    main()