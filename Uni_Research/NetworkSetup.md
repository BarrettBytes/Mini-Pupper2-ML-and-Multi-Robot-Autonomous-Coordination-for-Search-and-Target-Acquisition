# Mini Pupper Wi-Fi Setup

This guide will walk you through the steps to connect your Mini Pupper to a Wi-Fi network using the command line.

## Important Note for VMs
if you are using a VM to connect to the robot, ROS will not work if you are not on the same network and the network 
must be BRIDGED otherwise it will fail to send information over ROS

## Manual setup stage (each time you are connecting to a new wifi)
The following steps must be done manually on the robot each time you are using a new internet network, these need to be done on the robot using a HDMI and keyboard attached directly to the robot

## Steps

1. Open the network configuration file using nano:

   ```
   sudo nano /etc/netplan/50-cloud-init.yaml
   ```

2. In the nano editor, configure your Wi-Fi connection details using the following layout:

   ```yaml
   network:
     ethernets:
       eth0:
         dhcp4: true
         optional: true
     wifis:
       wlan0:
         access-points:
           "WiFi-9BNXD":
             password: "AAAA035$58"
           "Trusted_Autonomous_Sys_Wireless":
             password: "macl2022"
         dhcp4: true
         optional: true
     bridges:
       br0:
         addresses: [10.0.0.10/24]
         parameters:
           stp: true
           forward-delay: 4
         dhcp4: false
         optional: true
   version: 2
   ```

   Replace `WiFi-9BNXD` and `Trusted_Autonomous_Sys_Wireless` with your Wi-Fi network names, and replace `AAAA035$58` and `macl2022` with the corresponding Wi-Fi passwords.

   The important parts in this configuration are:

   - `wifis` section defines the Wi-Fi networks to connect to.
   - Each network is specified under `access-points` with its SSID and password.
   - `dhcp4: true` enables automatic IP address assignment for Wi-Fi.
   - `bridges` section defines a network bridge.
   - The bridge is configured with a static IP address `10.0.0.10/24`.
   - `parameters` specify the bridge settings, such as enabling Spanning Tree Protocol (STP) and setting the forward delay.
   - `optional: true` allows the system to continue booting even if the network is not available.

3. Save the changes and exit nano by pressing `Ctrl+X`, then `Y`, and finally `Enter`.

4. Generate the network configuration:

   ```
   sudo netplan generate
   ```

5. Apply the network configuration:

   ```
   sudo netplan apply
   ```

6. Start the Wi-Fi supplicant:

   ```
   sudo systemctl start wpa_supplicant
   ```

## SSH from a terminal
After this is done you can now use a terminal (such as your VM terminal) to SSH into the robot. If these steps were completed successfully the ip will appear on the display of the robot. 

In the terminal you can then type
  ```
   ssh ubuntu@<Ip address of mini pupper you read of the display>
  ```
This will allow you to control the dog with the terminal on a different device but cannot be done until the manual steps before were completed using the HDMI and keyboard attached directly to the robot

## Troubleshooting

If the network is not found or the connection is not established, try the following:

- Turn the robot off and on again.

- Run the following command to restart the Wi-Fi supplicant:

  ```
  sudo systemctl restart wpa_supplicant
  ```

- Check the status of the Wi-Fi supplicant:

  ```
  sudo systemctl status wpa_supplicant
  ```

If the issue persists, double-check your Wi-Fi configuration in the `50-cloud-init.yaml` file and ensure that the Wi-Fi network names and passwords are correct.

also make sure you do this on both the computer and mini pupper 
  ```
nano ~/.bashrc
  ```
  ```
export ROS_DOMAIN_ID=42 #add to the final line of the file
  ```
