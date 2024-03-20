# curve-based-3d-reconstruction

- src
- real_data
- simulation_data
- demo
  - 1_label.ipynb
    input: images; camera parameters; polygon or point cloud
    output: labels and occlusions
  - 2_reconstruction.ipynb
    input: images; camera parameters; labels; support threshold
    output: supported curve fragments
  - 3_B-spline.ipynb
    input: supported curve fragments
    output: marged curves
  - 4_visualization.ipynb
    input: marged curves
    outout: none
  - data

## License
The source code is licensed under the Apache License, Version2.0. The documents and data are licensed under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) License.
