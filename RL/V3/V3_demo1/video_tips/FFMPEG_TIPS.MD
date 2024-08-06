Remove static frames: ffmpeg -i simulation_6x6_2cats_2dogs_2needed.mp4 -vf "mpdecimate,setpts=N/FRAME_RATE/TB" -an output_no_static_frames.mp4
