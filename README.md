# curve_sketch
curve_sketch

 -src
 
 -real_data #experimental
 
 -simulation_data #experimental
 
 -demo

   -1_label.ipynb
    
    input: images; camera parameters; polygon or point cloud
    
    output: labels and occlusions
  
   -2_reconstruction.ipynb
    
    input: images; camera parameters; labels; support threshold
    
    output: supported curve fragments
  
   -3_B-spline.ipynb
    
    input: supported curve fragments
    
    output: marged curves
  
   -4_visualization.ipynb
    
    input: marged curves
    
    outout: none
  
   -data
 
