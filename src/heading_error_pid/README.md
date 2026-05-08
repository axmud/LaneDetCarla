# Codes

You can run ```pygame_controller.py``` freely. For using ```pid_tuning.py``` you first have to run ```carla_sysid.py```, then using generated files you can run it by using these lines on the command line terminal:

```cmd
python src\heading_error_pid\pid_tuning.py --lon src\system_identificator\sysid_longitudinal.csv --lat src\system_identificator\sysid_lateral.csv --meta src\system_identificator\sysid_meta.csv --wn-lat 2.0 --wn-lon 1.0
```

for drawing the result in the carla:

```cmd
python src\draw_trajectory_in_carla.py --log src\heading_error_pid\ego_trajectory.csv
```

for drawing the results:

```cmd
python src\plot_trajectory.py --logs src\heading_error_pid\ego_trajectory_heading.csv
```
