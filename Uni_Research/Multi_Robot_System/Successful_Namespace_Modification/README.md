
## Multi-Robot System Namespace Investigation
It's working but only tested with one robot
---

ALL OF THIS IS DONE ON THE ROBOT NOT COMPUTER

This section outlines the changes needed to implement namespace adjustments across multiple ROS2 nodes for the Mini Pupper robots.

1. **Namespace Adjustment for All Nodes**
   - Make the following changes, ensuring that backups of the original files are stored for reference.

---

### Steps to Modify Code

1. **Modify `bringup_basic_remap.launch.py`:**
   - Open the file in an editor:
     ```bash
     sudo nano ~/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup/launch/bringup_basic_remap.launch.py
     ```

2. **Modify `bringup_launch_remap.py`:** using the champ_bringup_launch_remapp.py file
   - Edit using:
     ```bash
     sudo nano ~/ros2_ws/install/champ_bringup/share/champ_bringup/launch/bringup_launch_remapp.py
     ```

3. **Modify `modified_hardware_interface.py`:**
   - Open the file:
     ```bash
     sudo nano /home/ubuntu/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup/launch/modified_hardware_interface.py
     ```

4. **Modify `modified_servo_interface.py`:**
   - Edit with:
     ```bash
     sudo nano /home/ubuntu/ros2_ws/src/mini_pupper_ros/mini_pupper_driver/launch/modified_servo_interface.py
     ```

5. **Modify `modified_lidar_ld06.py`:**
   - Open the file:
     ```bash
     sudo nano /home/ubuntu/ros2_ws/src/mini_pupper_ros/mini_pupper_driver/launch/modified_lidar_ld06.py
     ```

---

We need to make some further changes however these steps will require changing existing files instead of making new ones so backup will be required

### Modification with Backup Procedures:

1. **Backup `mini_pupper_description.launch.py`:**
   - We have copied the original content of `mini_pupper_description.launch.py` and stored it in the repository under the name `original_mini_pupper_description.py`.
   - Modify the file by copying the content from `modified_mini_pupper_description.py`:
     ```bash
     sudo nano /home/ubuntu/ros2_ws/src/mini_pupper_ros/mini_pupper_description/launch/mini_pupper_description.launch.py
     ```

2. **Backup `champ_description/launch/description.launch.py`:**
   - We have copied the original content of `champ_description/launch/description.launch.py` and stored it in the repository under the name `original_champ_description_launch.py`.
   - Modify the file by copying the content from `modified_champ_description_extra_frameID_changes.py`:
     ```bash
     sudo nano /home/ubuntu/ros2_ws/src/champ/champ/champ_description/launch/description.launch.py
     ```

3. **Backup `/mini_pupper_driver/servo_interface.py`:**
   - We have copied the original content of `/mini_pupper_driver/servo_interface.py` and stored it in the repository under the name `original_driver_servo_interface.py`.
   - Modify the file by copying the content from `modified_driver_servo_interface.py`:
     ```bash
     sudo nano /home/ubuntu/ros2_ws/src/mini_pupper_ros/mini_pupper_driver/mini_pupper_driver/ servo_interface.py
     ```
      i dont know why there is a space between here "/ servo_interface.py" seems to be a mistake they made but its how it is on mine lol so i assume its like this on yours
   
      if this doesnt work try
      ```bash
      cd  /home/ubuntu/ros2_ws/src/mini_pupper_ros/mini_pupper_driver/mini_pupper_driver/  
      ls
      ```
        
      if servo_interface.py is there
      type
      ```bash
      sudo nano servo_interface.py
      ```
---

### Testing the Code

- Test with (this is the only step you can do on the computer if you wish):
  ```bash
  . ~/ros2_ws/install/setup.bash
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/robotx/cmd_vel
  ```

- Launch file path for testing:
  ```bash
  . ~/ros2_ws/install/setup.bash
  ros2 launch ~/ros2_ws/src/mini_pupper_ros/mini_pupper_bringup/launch/bringup_basic_remap.launch.py
  ```

---

### CAN IGNORE THIS Package Information
- We have copied the original content of `package.xml` from `mini_pupper_description` and stored it in the repository under the name `original_package.xml`:
  ```bash
  ~/ros2_ws/src/mini_pupper_ros/mini_pupper_description/package.xml
  ```

---

