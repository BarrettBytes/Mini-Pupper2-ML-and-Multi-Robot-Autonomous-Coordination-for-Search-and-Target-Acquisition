# Machine Learning and Multi-Robot Autonomous Coordination for Search and Target Acquisition Using Quadruped Robots

This repository contains the code for a quadruped robot research project. The primary goal is the design, implementation, and testing of a low-cost, autonomous quadruped robot for Search, Track, and Acquire (STA) operations. By integrating mechanical, electrical, and computer vision systems with advanced machine learning techniques, this work aims to demonstrate the feasibility of deploying affordable autonomous quadrupeds in real-world scenarios.

## Key Components

### Objectives
1. **Mechanical, Electrical, and Vision System Integration:**  
   Ensure robust hardware capabilities for locomotion and object detection.

2. **Advanced Machine Learning:**  
   Train the quadruped using Behavioral Cloning (BC), Curriculum Learning (CL), Reinforcement Learning (RL), and Inverse Reinforcement Learning (IRL).

3. **STA Operation Execution:**  
   Successfully deploy quadrupeds in both simulated and physical STA scenarios.

4. **Autonomous Multi-Robot Coordination:**  
   Enable communication and coordination among multiple robots for efficient STA operations.

### Methods
- **Hardware Integration:**  
  Utilized an OAK-D camera for object detection, LiDAR for distance measurement, and servo motors for locomotion.

- **Software Integration:**  
  Implemented the Robot Operating System 2 (ROS 2) for modular communication, control, and data handling.

- **Machine Learning Models:**  
  Developed neural networks using MobileNet-SSD v2 for object detection.  
  Employed RL techniques, along with IRL, for planning and decision-making.

## Key Results

- **See Recording:**  
https://youtu.be/EaRwWQGmjD0

- **Simulation Testing:**  
  The quadrupeds demonstrated successful navigation in simulated environments, identifying targets, avoiding hazards, and adapting to changing conditions.

- **Laboratory Testing:**  
  Physical experiments showcased the robot’s ability to detect and acquire targets, though mechanical challenges (e.g., terrain irregularities) and communication reliability were identified as areas needing improvement.

- **Limitations Identified:**  
  Future refinements are needed in terrain handling, multi-robot communication robustness, and the computational efficiency of machine learning models.

## Experimentation
Although the full project report is not publicly available at this stage, the code and system configurations are open for experimentation. Users are encouraged to explore, adapt, and refine the methodologies to suit their own needs. This project provides a foundational framework for further innovation and discovery in affordable autonomous robotics.

## Recommendations for Future Work
1. Enhance terrain-handling capabilities with improved balancing algorithms.  
2. Allocate increased computational resources for more complex and robust neural networks.  
3. Improve proximity detection and communication strategies for better multi-robot coordination.  
4. Explore alternative hardware configurations and refine vision algorithms to increase system robustness.

## Conclusion
This project demonstrates the feasibility of deploying low-cost quadruped robots for STA operations. Through the integration of machine learning, mechanical design, sensor integration, and multi-robot coordination, this work illustrates the potential for affordable and effective autonomous robotic solutions.

---

# Repository Structure and Instructions

This repository includes code for both single-robot IRL training and Multi-Robot System (MRS) integration.

## Code Structure
- **Inverse Reinforcement Learning (IRL):**  
  Located in `Uni_Research/IRL`. IRL enables the quadruped to learn from expert demonstrations rather than relying purely on programmed reward structures.

- **Multi-Robot System (MRS):**  
  Found in `Uni_Research/Multi_Robot_System/IRL`. This component deals with coordinating and controlling multiple quadruped robots, including inference scripts and related modules for executing multi-robot STA operations.

## Setup and Usage
Please refer to the following `README.md` files for detailed setup instructions:

1. **Main Setup Instructions:**  
   [README.md](Uni_Research/README.md)  
   Follow these instructions first to set up the environment and dependencies.

2. **Multi-Robot System Setup Instructions:**
   [Multi Robot Namespace README.md](Uni_Research/Multi_Robot_System/Successful_Namespace_Modification/README.md)
   [Multi Robot IRL README.md](Uni_Research/Multi_Robot_System/IRL/README.md)
   Follow these instructions to set up and run the multi-robot inference and coordination system.

## Key Files
- **Training Code (Single Robot):**  
  [Demo3.py](Uni_Research/IRL/Demo3.py)  
  Main code for training the quadruped robot using IRL techniques.

- **Inference Code (Multi-Robot):**  
  [Multi_Robot_inference_counts_to_three.py](Uni_Research/Multi_Robot_System/IRL/Multi_Robot_inference_counts_to_three.py)  
  Main code for running inference on the multi-robot system, enabling coordinated decision-making and task execution.

## Contributions and Testing
Due to limited access to physical robots after project completion, extensive refactoring and testing of proposed changes is challenging. Users are welcome to open issues and submit pull requests. Changes to training code can be tested offline, while changes requiring physical robot testing are more difficult to verify.

---

**Feel free to open issues or submit pull requests.** This project serves as a starting point for others to build upon, advancing affordable, autonomous quadruped robotics.

---

## Repository Contributors
Khiya Barrett 
Cho Ting Lee
Muhammad Suhaib Rehan


## Do Whatever You Want With The Code :)
If you are feeling nice and kind then link to this repo somewhere in your project if you use it. But you aren't obligated.
You could even give us a star and suggest code improvements if your in a really good mood :p
