# RT entry and Graph Search

## build and run

```bash
mkdir build
cd build
cmake ..
make
cd ..
cd bin
./test
```

## parameter
There are two kinds of parameters. One is runtime parameter, which can be specified in `flags.cpp` or run command. The other is the macro definition parameter, which needs to modify the macro definition in `head.h` or `graph_search.cuh`.

### runtime parameter
1. Modify flags.cpp directly like
```bash
DEFINE_string(data_path, "/data/myl/sift1M/sift1M_base.fvecs", "path of datasets");
```
If you do this, you need to compile and run again.
2. Specify the value of the parameter at run time
```bash
./test --data_path=/data/myl/sift1M/sift1M_base.fvecs --query_path=/data/myl/sift1M/sift1M_query.fvecs
```

### Macro definition parameter
change in source file directly, compile and run.

## Parameter interpretation
1. max_hits: When using RT for intersection, at most max_hits AABB can be saved.
2. expand_ratio: The extent to which each actual AABB scales out when building a scenario.
3. point_ratio: When dividing kd trees, each leaf node (AABB) can contain a maximum of `num_of_points x point_ratio` points
4. DIM、ENTRY_DIM、GRAPH_DIM：The actual saved vector dimension; The vector dimension used when calculating entry points; The vector dimension used in graph search
5. REORDER: Whether to reorder the image after graph search