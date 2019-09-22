# Multiple-view geometry
This includes 3 main functions:
* Reconstruct 3D model from images using structure from motion algorithm,
* Perform camera pose estimation with known 3D model using direct matching algorithm,
* Dense 3D recontruction using Patch-Based Multi-View Stereo algorithm.

## Build project
Build project with cmake:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Run project
Clone and copy test data to build folder:
```
cd ../../
git clone https://github.com/nqoptik/computer_vision_data.git
cd multiple_view_geometry/build/
cp -r ../../computer_vision_data/multiple_view_geometry/build/* .
```

Create output folders:
```
mkdir img_out
mkdir log_img
mkdir output
```

Run structure form motion:
```
./structure_from_motion
```

Run pose estimation:
```
./pose_estimation
```

Run pmvs:
```
./pmvs
```
