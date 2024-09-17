make vid from frames: 

ffmpeg -framerate 10 -i frames%05d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p -y output.mp4

Remove static frames: 

ffmpeg -i simulation_6x6_2cats_2dogs_2needed.mp4 -vf "mpdecimate,setpts=N/FRAME_RATE/TB" -an output_no_static_frames.mp4
