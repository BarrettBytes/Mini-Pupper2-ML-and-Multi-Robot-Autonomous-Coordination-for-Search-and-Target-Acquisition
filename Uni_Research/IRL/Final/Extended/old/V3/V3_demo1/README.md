** updates logged newest to oldest**

** Updated inference: Launch_mobile_pure_inference_aggressive.py **

instead of Launch_mobile_pure_inference.py upgrade to Launch_mobile_pure_inference_aggressive.py it should run a lot better

*updated for V3* 

Your gonna have to run monitor_and_restart_mobile.py on the mini pupper (save it in the mini pupper using nano), you will need to put PASCAL.names in the same folder on the mini pupper. 



(you will have also had to do this stuff: https://github.com/BarrettBytes/QuadropedRobotHonoursProject/blob/main/OpenCV_Camera_Robot_Intergration_Files/ReadMe.md)  On the mini pupper you will also have to run mini pupper bringup

see the readme for these but these are the sudo nanos -> u will need to do all the steps on the readme above not just these:

sudo nano /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py

sudo nano /opt/ros/humble/share/depthai_examples/launch/PASCAL.names


on the computer you can save a moddel with save_training_run.py and then run with Launch_mobile_pure_inference.py. Alternatively just use the saved 0.9_policy.pth model (it is set to this model by default if you want a new model, in the Launch_mobile_pure_inference.py code do a control F for  0.9_policy.pth and replace it with the one you saved). Note there is code to automatically catch the cats and avoid the dogs using hard coding you can test using Launch_mobile_auto_catch.py instead of Launch_mobile_pure_inference.py. (run with python3 Launch_mobile_pure_inference.py)

to run:
in one ssh'd terminal to the pupper:

cd /opt/ros/humble/share/depthai_examples/launch/
python3 /opt/ros/humble/share/depthai_examples/launch/monitor_and_restart_mobile.py

in another ssh'd terminal
. ~/ros2_ws/install/setup.bash # setup.zsh if you use zsh instead of bash
ros2 launch mini_pupper_bringup bringup.launch.py

in the computer
where ever u put the Launch_mobile_pure_inference.py or whichever launch you have run
python3 Launch_mobile_pure_inference.py
or whatever one

**notes**
for Demo3.py you may need to: sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
as well as pip installing all that needs to be pip installed (remember cv2 is opencv-python)

