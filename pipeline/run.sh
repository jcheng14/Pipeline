#!/bin/bash

# Wait for 3 hours (10800 seconds)
# echo "The script will start in 3 hours..."
# sleep 10800

echo "Start SAM"

# Run your Python script
echo "Starting the Python script..."
echo "Test_0312/back/original/left"
python3 ../scripts/amg.py --input /home/jamesxyye/Data/RainerVeneer/Test_0312/back/original/left --output /home/jamesxyye/Data/RainerVeneer/Test_0312/back/seg_result/left --model-type vit_h --checkpoint weights/sam_vit_h_4b8939.pth 

echo "Test_0312/back/original/top"
python3 ../scripts/amg.py --input /home/jamesxyye/Data/RainerVeneer/Test_0312/back/original/top --output /home/jamesxyye/Data/RainerVeneer/Test_0312/back/seg_result/top --model-type vit_h --checkpoint weights/sam_vit_h_4b8939.pth 

echo "Test_0312/back/original/right"
python3 ../scripts/amg.py --input /home/jamesxyye/Data/RainerVeneer/Test_0312/back/original/right --output /home/jamesxyye/Data/RainerVeneer/Test_0312/back/seg_result/right --model-type vit_h --checkpoint weights/sam_vit_h_4b8939.pth 


echo "Generate Masked Image"
python3 generate_masked_image.py --data /home/jamesxyye/Data/RainerVeneer/Test_0312/back/left --seg_result /home/jamesxyye/Data/RainerVeneer/Test_0312/back/seg_result/left --output /home/jamesxyye/Data/RainerVeneer/Test_0312/back/masked_date/left
python3 generate_masked_image.py --data /home/jamesxyye/Data/RainerVeneer/Test_0312/back/top --seg_result /home/jamesxyye/Data/RainerVeneer/Test_0312/back/seg_result/top --output /home/jamesxyye/Data/RainerVeneer/Test_0312/back/masked_date/top
python3 generate_masked_image.py --data /home/jamesxyye/Data/RainerVeneer/Test_0312/back/right --seg_result /home/jamesxyye/Data/RainerVeneer/Test_0312/back/seg_result/right --output /home/jamesxyye/Data/RainerVeneer/Test_0312/back/masked_date/right


echo "Process Point Cloud"
python3 process_point_cloud.py --input /home/jamesxyye/Data/RainerVeneer/Test_0312/back/point_cloud.ply --output_dir /home/jamesxyye/Data/RainerVeneer/Test_0312/back
