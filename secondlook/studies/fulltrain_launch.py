"""E42 launcher: run an unchanged pipeline module with the full-TRAIN-pool tasks installed. Usage: python -m secondlook.studies.fulltrain_launch <module> [module args...]"""
import runpy
import sys

from secondlook.data.tasks_fulltrain import install

if __name__ == '__main__':
    install()
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name='__main__', alter_sys=True)
