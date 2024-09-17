# Time Synchronization for Mini Pupper

Synchronizing the system clocks on the controlling computer and the Mini Pupper is crucial for the proper functioning of the Robot Operating System (ROS) and the Mini Pupper's navigation and SLAM capabilities. Failure to synchronize the clocks can lead to issues with message timing and cause navigation and SLAM failures.

## Installing Chrony

Install Chrony on both the controlling computer and the Mini Pupper:

```bash
sudo apt install chrony
```

## Checking Chrony Status

Check the status of the `chronyd` daemon:

```bash
sudo systemctl status chronyd
```

Start `chronyd` if it's not running:

```bash
sudo systemctl start chronyd
```

## Verifying Synchronization

Verify if Chrony is properly synchronized:

```bash
chronyc tracking
```

View information about Chrony's time sources:

```bash
chronyc sources
```

To stop Chrony:

```bash
sudo systemctl stop chronyd
```

Ensure these steps are performed on both the controlling computer and the Mini Pupper to guarantee accurate clock synchronization and proper ROS functionality.

Apologies for the confusion. Here's the corrected markdown:

# Setting up Chrony Controller for LAN Networks

This guide explains how to set up one computer as the controller (also known as the server or master) using `chronyc` for LAN networks.

## Prerequisites

- Install the `chrony` package on the controller computer if it's not already installed. On Ubuntu or Debian, you can use the following command:

  ```bash
  sudo apt install chrony
  ```

## Configuration Steps

1. Open the `chrony` configuration file (`/etc/chrony/chrony.conf`) using a text editor with root privileges. For example:

   ```bash
   sudo nano /etc/chrony/chrony.conf
   ```

2. In the configuration file, locate the `allow` directive. This directive specifies the network or IP addresses that are allowed to synchronize with the controller. Uncomment or add the `allow` directive followed by the network or IP addresses you want to allow. For example, to allow any client in the local network (192.168.0.0/24) to synchronize with the controller, add the following line:

   ```
   allow 192.168.0.0/24
   ```

   **Note:** If you are setting up `chrony` for a normal network (not a LAN), you should comment out the `allow` directive to prevent unauthorized access to your controller.

3. Save the changes to the configuration file and exit the text editor.

4. Restart the `chrony` service to apply the changes:

   ```bash
   sudo systemctl restart chrony
   ```

5. Verify that the `chrony` service is running:

   ```bash
   sudo systemctl status chrony
   ```

   You should see output indicating that the service is active and running.

   Now do the same on the client device except instead of allow do:

  ```bash
  server 192.168.0.100
  ```

7. To check the status of the `chrony` service and see the list of connected clients, you can use the `chronyc` command-line tool. Run the following command:

   ```bash
   chronyc clients
   ```

   This will display the list of connected clients and their synchronization status.

## Troubleshooting

If the other devices don't appear as clients when you run `chronyc clients` on the controller, try the following troubleshooting steps:

1. Check the firewall: Ensure that the firewall on the controller computer is allowing incoming NTP traffic (UDP port 123).

2. Verify network connectivity: Make sure that the client computers can reach the controller computer over the network.

3. Check the client configuration: Ensure that the client computers are properly configured to use the controller as their NTP server.

4. Restart the `chrony` service on the clients: After making any changes to the client configuration, restart the `chrony` service on the client computers.

5. Check the `chrony` logs: Examine the `chrony` logs on both the controller and the clients for any error messages or indications of issues.

6. Verify time synchronization: On the client computers, use the `chronyc tracking` command to check if they are successfully synchronizing with the controller.

Remember to give it some time (a few minutes) for the clients to establish a connection and synchronize with the controller. The synchronization process may take a while initially.

If you still face difficulties, consult the `chrony` documentation or seek further assistance from the `chrony` community or support channels.

## DON'T DO THIS: CAUSED ISSUES, WILL FIX INSTRUCTIONS WHEN I FIND THE SOLUTIONS: Installing DDS 
you need to install DDS on all ROS environments as well (your computer and the mini pupper)

install eclipse cyclose DDS

```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
```

set the environmental variables
add this to
```bash
nano ~/.bashrc
```
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```
then run
```bash
source ~/.bashrc
```

after u have set up ur computer with a static IP (lan only) then on the robot goto

sudo nano /etc/chrony/chrony.conf

and write

server 192.168.0.103 prefer iburst


