# DIY Eye Tracker

Tracks where the user is looking! This setup has 2 cameras on a Pi: one camera facing the eye, the other facing the world.
For my personal results with this, see my [engineering portfolio](https://sites.google.com/view/rberkun/personal-projects?authuser=0).

This is inspired by and builds off of [Jason Orlosky's](https://github.com/JEOresearch/EyeTracker) work. Specifically, I used OsloskyEyeTrackerLite for pupil detection, because the 3D gaze vector thing wasn't working out for me + I needed something more lightweight for streaming. I don't specialize in computer vision so I vibe coded the whole project ([as suggested by Orlosky himself!]). (https://www.jeoresearch.com/eyetracking.html#:~:text=Step%208%3A%20Use%20AI%20to%20build%20your%20own%20gaze%20application)).  

## Materials + Construction
### Materials
| Purpose | Item | Item link | Item cost (2026)|
| --- | --- | --- | --- |
| Power for Pi | Miady USB C Mini Portable Charger | [Amazon](https://www.amazon.com/dp/B0FC2NK1YR?lv=shuf&redirect=true&smid=A2NKEN9O69YXYL&channelId=500&ref_=ox_sc_act_title_3&plpRedirect=mhFallback&th=1) | $14 |
| Power extension cable | USB C Extension Cable 0.25m (Male C input --> Male C output) | [Amazon](https://www.amazon.com/dp/B0D1TYYVR3?lv=shuf&redirect=true&smid=A2SBTRGJKT6RTQ&channelId=500&ref_=ox_sc_act_title_2&plpRedirect=mhFallback&th=1) | $10 |
| Adapter for Pi power port | JMOX USB C to Micro Adapter (Male C input --> Male micro output) | [Amazon](https://www.amazon.com/dp/B07GH5KJH2?ref=ppx_yo2ov_dt_b_fed_asin_title&th=1) | $6 |
| Eye-facing camera + cable | Raspberry Pi Camera Module 3 Wide NoIR - 12MP 120 Degree - Wide Angle Infrared Lens + INCLUDE Raspberry Pi Zero FPC Camera Cable | [Adafruit](https://www.adafruit.com/product/5660) | $42 |
| World-facing camera | Innomaker U20CAM-720P UVC Camera with 120 DFOV | [Amazon](https://www.amazon.com/dp/B0CLRJZG8D) | $16 |
| Adapter for world-facing camera | USB 2.0 to micro adapter (Male USB A input --> Male micro output) | [Amazon](https://www.amazon.com/dp/B01C6032G0?ref_=ppx_hzsearch_conn_dt_b_fed_asin_title_3) | $5 |
| Pi for camera streaming | Raspberry Pi Zero 2W | [Raspberry Pi](https://www.raspberrypi.com/products/raspberry-pi-zero-2-w/) | $15 |
| Memory card for Pi | Micro SD card (64GB+) | [Amazon](https://www.amazon.com/dp/B08TJRVWV1?lv=shuf&channelId=500&plpRedirect=mhFallback&th=1) | $12 |
| Illuminate pupil from iris | IR led 3mm 940nm | [Amazon](https://www.amazon.com/dp/B0DZ2KSGZ4?s=bazaar&ref_=ppx_hzsearch_conn_dt_b_amznbzr_ppx_yo_asin_title_2) | $4 |

### Construction
- #### Wiring
  - **Power**: Portable charger --> power extension cable --> USB C to micro adapter --> USB micro power port on Pi (this is the port closer to the CSI port)
  - **Eye-facing camera**: NoIR Camera --> FPC cable --> CSI-2 port on Pi
  - **World-facing camera**: USB Camera --> USB cable --> USB A to micro adapter --> USB micro data port on Pi (this is the port farther from the CSI port)
  - **IR led**: 5V on Pi --> 470 ohm resistor --> IR led --> GND on Pi
- #### Software setup
  - Use the Rasberry Pi imager to flash your SD card with Rasberry Pi OS Lite (32 bit) + set the Pi up on the same Wifi as your laptop
  - **Make sure you are on a 2.4GHz network as the Pi Zero 2W can't connect to 5G networks** (I had to use a Wifi extender to get a dedicated 2.4GHz channel for my home wifi)
  - SSH into the Pi and install all the needed packages:
    ```
    sudo apt update
    sudo apt full-upgrade -y
    
    sudo apt install -y \
    python3-picamera2 \
    python3-opencv \
    python3-flask \
    python3-numpy \
    python3-zmq
    ```
  - Copy everything in `pi_files` onto your Pi
  - You can find your pi's IP address by typing `hostname -I` on your Pi. Update the variable `PI_IP` and `PI_STREAM` in each file in `calibate_and_run` with this IP address.
- #### Mechanical setup
  -   **Camera mounting:** I took an empty glasses frame (which I 3D printed, something I regret as this was very flimsy) and hot-glued the world-camera to the top of the frame. I mounted the world-facing camera upside-down so that the cable would point upwards. Then I mounted the eye-camera a few inches from my non-dominant eye (my left eye).
  -   **Pi+power mounting:** I zip-tied the power cable and power bank to a headband
     
## How to Run
- **Crop the eye image**: For me, my eye camera shows a lot of things that are not my eye. So I crop the video to be smaller to focus on my eye and use less WiFi bandwidth.
  - On the Pi: Run `python eye_camera_stream.py`. It will prompt you to put in 4 numbers, but just leave it empty and press ENTER for now.
  - On your laptop: Run `select_ROI.py`. This will show you a picture of what the eye-facing camera sees. Click and drag a box region around your eyeball. 
  - After closing the `select_ROI.py` script, 4 numbers will have been printed to terminal. Save these four numbers.
  - On the pi: Close `python eye_camera_stream.py` and re-run it using the 4 numbers from terminal
  - Sanity check: On the laptop, run `Orlosky3DEyeTrackerLite.py`, and make sure it's streaming a cropped picture of your eyeball with a green circle around your pupil.
  - Finish: Close all scripts on the Pi and laptop
- **Calibrate**:
  - On the Pi: Run `dual_camera_stream_v2.py`. Input the same 4 numbers you used in the "Crop eye image step"
  - On the laptop: Run `scene_calibrate_v3.py`.
    - Two windows should pop up, one with a video of your eye, the other with what your world-facing camera sees. Make sure these are as expected.
    - Adjust your head so that your computer screen takes up most of your desired field of view. By "desired field of view," I mean you should calibrate this to what your desired use case is. For example, if I'm planning to play piano, I'll put my laptop low in my lap to mimck how I'll be looking down at my hands. If I'm going to be doing something on my laptop, I'll try to sit closer to the laptop on my desk.
    - Press space while the world-facing camera stream is selected to start the calibration process.
    - The calibration process shows you a grid of dots on your computer screen. You should fixate on each dot as it is shown. When the dot is pink, calibration is not being done, and you can use this time to move your eyes/blink. When the dot is green, fixate your eyes on the dot and don't move.
    - When calibration finishes, check the files in `calibration_debug`. Check that for each dot, the script correctly identified where the dot is on the computer screen as well as reasonably labeled your pupil location. (It is ok if a few dots are "skipped" and did not make the calibration. If a ton of them are skipped, try changing your lighting -- I've found that having a lamp right slightly above + behind the laptop helps).
    - If your calibration results are bad you can calibrate again, but beware re-running the calibration script deletes all your old files.
- **Record**:
  - Run `dual_eye_tracker_gui_v3.py` and you should see a stream of your eye + the world camera side by side. Where you are looking is marked on the world camera stream with a red dot.
  - Theoretically the GUI should be able to record videos, in practice the timing of the videos is really off for some reason. So I use a built-in Windows screen recorder of the stream to take my data.
