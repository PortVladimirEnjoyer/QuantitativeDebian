# QuantOS

A custom OS based on Debian with a minimalist XFCE environment.



![QuantOS Screenshot](https://github.com/justworking505/QuantitativeDebian/blob/main/desktoppng)

![QuantOS Screenshot](https://github.com/justworking505/QuantitativeDebian/blob/main/screenshot1)


## Description

This is a custom OS made with a standard Debian ISO, to which I've installed a minimalist XFCE package. The system comes with just a file manager and Firefox pre-installed.

The "QuantOS" system has the following structure:
![image](https://github.com/user-attachments/assets/2291346e-43a9-4206-936a-10430a4d3b60)


## Features

- Minimalist Debian-based OS
- XFCE desktop environment
- Pre-installed file manager
- Pre-installed Firefox browser
- Includes QuantOS system components

## Notes

- This is the first version - more programs may be added in the future if needed
- In `llm.py`, the script automatically installs the wizard-math model if it's not present
- The model is not included by default in the OS because:
  - It's quite large in size
  - Not strictly necessary for basic functionality

## Installation


To install QuantOS, simply:

1. Download the ISO file from [releases/latest]
2. Flash it to a USB drive using a tool like:
   - [Rufus](https://rufus.ie/) (Windows)
   - [Balena Etcher](https://www.balena.io/etcher/) (Windows/macOS/Linux)
   - `dd` command (Linux)
3. Boot from the USB drive (enter your BIOS/UEFI boot menu)
   
*Alternatively, for testing purposes, you can run QuantOS in a virtual machine (recommended for first-time users):*
- Use VirtualBox, VMware, or QEMU
- Create a new VM and select the QuantOS ISO as the boot media


