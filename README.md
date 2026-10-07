# pyrxmesh_parameterization


Performance Comparison: Running each test 5 times yields the following output between pyrxmesh and rxmesh. The time measured is the time in milliseconds that rxmesh or pyrxmesh took to perform stochastic gradient descent 

GPU: RTX 5070

## PyRXMesh

bunnyhead.obj

command: python run.py --obj-file-name=bunnyhead.obj --learning-rate=1e-9 --num-iter=100 --polyscope=0
```
iterations = 100, energy = 3145.810547 -> 2169.919922, time = 6.522528 ms
iterations = 100, energy = 3145.810303 -> 2169.919678, time = 6.891808 ms
iterations = 100, energy = 3145.809570 -> 2169.921875, time = 6.939456 ms
iterations = 100, energy = 3145.809570 -> 2169.921631, time = 6.955488 ms
iterations = 100, energy = 3145.810547 -> 2169.919678, time = 6.838656 ms
```
armadillo_cut_high

command: python run.py --obj-file-name=armadillo_cut_high.obj --uv-file-name=armadillo_cut_high_uv.obj --learning-rate=1e-15 --num-iter=100 --polyscope=0
```
iterations = 100, energy = 55968.761719 -> 45667.226562, time = 20.283424 ms
iterations = 100, energy = 55968.769531 -> 45667.230469, time = 20.591743 ms
iterations = 100, energy = 55968.777344 -> 45667.230469, time = 19.877792 ms
iterations = 100, energy = 55968.765625 -> 45667.222656, time = 20.516865 ms
iterations = 100, energy = 55968.765625 -> 45667.230469, time = 20.791264 ms
```
camel_head

command: python run.py --obj-file-name=camel_head.obj --learning-rate=1e-9 --num-iter=100 --polyscope=0
```
iterations = 100, energy = 67.383957 -> 66.437309, time = 6.609120 ms
iterations = 100, energy = 67.383972 -> 66.436981, time = 6.612768 ms
iterations = 100, energy = 67.383972 -> 66.437912, time = 7.226720 ms
iterations = 100, energy = 67.383995 -> 66.436836, time = 6.827936 ms
iterations = 100, energy = 67.384003 -> 66.436920, time = 6.827040 ms
```

## RXMesh

bunnyhead.obj

run command: param.exe --input=C:/dev/pyrxmesh_parameterization/meshes/bunnyhead.obj --lr=1e-9 --iter=100

```
iterations= 100, energy= 3145.8494 -> 2168.0715, time= 4.823424 (ms)
iterations= 100, energy= 3145.8618 -> 2168.0803, time= 3.9384 (ms)
iterations= 100, energy= 3145.8389 -> 2168.0713, time= 4.822848 (ms)
iterations= 100, energy= 3145.84 -> 2168.0696, time= 4.832 (ms)
iterations= 100, energy= 3145.8389 -> 2168.0693, time= 4.677568 (ms)
```


armadillo_cut_high

run command: param.exe --input=C:/dev/pyrxmesh_parameterization/meshes/armadillo_cut_high.obj --uv=C:/dev/pyrxmesh_parameterization/input_uvs/armadillo_cut_high_uv.obj --lr=1e-15 --iter=100

```
iterations= 100, energy= 55968.766 -> 45667.227, time= 12.898816 (ms)
iterations= 100, energy= 55968.766 -> 45667.227, time= 12.964448 (ms)
iterations= 100, energy= 55968.76 -> 45667.227, time= 13.216096 (ms)
iterations= 100, energy= 55968.76 -> 45667.22, time= 12.855968 (ms)
iterations= 100, energy= 55968.76 -> 45667.223, time= 12.913408 (ms)
```
camel_head

run command: param.exe --input=C:/dev/pyrxmesh_parameterization/meshes/camel_head.obj --lr=1e-9 --iter=100
```
iterations= 100, energy= 67.39735 -> 66.44523, time= 4.387904 (ms)
iterations= 100, energy= 67.398 -> 66.4454, time= 4.973504 (ms)
iterations= 100, energy= 67.39731 -> 66.444695, time= 4.448192 (ms)
iterations= 100, energy= 67.39898 -> 66.445526, time= 4.878176 (ms)
iterations= 100, energy= 67.39716 -> 66.44486, time= 4.611264 (ms)
```