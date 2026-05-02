# Short description of the folder

There are bunch of code files in here. ```helper.py``` is essential for detecting the lanes and is used by the codes which use lane detection. ```myFirstCode.py``` is just a piece of code by myself when I was a noob😆. ```pygame_controller.py``` is the main code which I developed my control techniques inside. In the ```system_identification``` folder, there are codes for getting optimal PID parameters for the controller. If you want to create your own PID parameters first run ```carla_sysid.py```, then this line in this file folder:

```cmd
python pid_tuning.py --lon sysid_longitudinal.csv --lat sysid_lateral.csv --meta sysid_meta.csv --wn-lat 1.5 --wn-lon 1.0 --target-speed 30
```
