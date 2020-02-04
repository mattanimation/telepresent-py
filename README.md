# Telepresent Python

This is a client to act as a peer connecting to the [Telepresent](https://www.github.com/mattanimation/Telepresent) server written in python.

**Note:** This is still under active development and isn't quite ready for external use, however it should function out of the box if you have a server up and running.

## Requirements
* Python3.5 and up
* an instance of the `SignalServer` and `Frontend` running on the web somewhere
* ideally a `coturn` server too

## Installation
`pip3 install -r requirements.txt`

## Useage
`python3 telepresent_client.py`


### TODO
[] - clean up of code
[] - switch to socket.io for more common interface to signal server?
[] - some computer vision features?
[] - ROS2 integration
