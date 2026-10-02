# pyrxmesh_parameterization


Performance Comparison: Running each test 5 times yields the following output between pyrxmesh and rxmesh. The time measured is the time in milliseconds that rxmesh or pyrxmesh took to perform stochastic gradient descent 

GPU: RTX 5070

## PyRXMesh

bunnyhead.obj

command: python run.py --obj-file-name=bunnyhead.obj --learning-rate=1e-9 --num-iter=100 --polyscope=0
```
iterations = 100, energy = 3145.810303 -> 2169.919678, time = 16.402300 ms
iterations = 100, energy = 3145.810547 -> 2169.920898, time = 15.579800 ms
iterations = 100, energy = 3145.808350 -> 2169.920898, time = 16.634000 ms
iterations = 100, energy = 3145.809326 -> 2169.921631, time = 15.097500 ms
iterations = 100, energy = 3145.809814 -> 2169.921631, time = 16.449700 ms
```
armadillo_cut_high

command: python run.py --obj-file-name=armadillo_cut_high.obj --learning-rate=1e-15 --num-iter=100 --polyscope=0
```
iterations = 100, energy = 54410.917969 -> 45229.300781, time = 21.219100 ms
iterations = 100, energy = 54406.000000 -> 45315.683594, time = 20.685500 ms
iterations = 100, energy = 54356.074219 -> 45287.746094, time = 21.624200 ms
iterations = 100, energy = 54442.101562 -> 45329.718750, time = 22.042000 ms
iterations = 100, energy = 54246.734375 -> 45265.785156, time = 21.768800 ms
```
camel_head

command: python run.py --obj-file-name=camel_head.obj --learning-rate=1e-9 --num-iter=100 --polyscope=0
```
iterations = 100, energy = 67.383987 -> 66.437096, time = 15.962100 ms
iterations = 100, energy = 67.383987 -> 66.437096, time = 15.713600 ms
iterations = 100, energy = 67.383934 -> 66.437759, time = 16.663200 ms
iterations = 100, energy = 67.384033 -> 66.437477, time = 16.402000 ms
iterations = 100, energy = 67.383980 -> 66.436974, time = 15.369400 ms
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

run command: param.exe --input=C:/dev/pyrxmesh_parameterization/meshes/armadillo_cut_high.obj --lr=1e-15 --iter=100

```

```
camel_head

run command: param.exe --input=C:/dev/pyrxmesh_parameterization/meshes/camel_head.obj --lr=1e-9 --iter=100
```

```