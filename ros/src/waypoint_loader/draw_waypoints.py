#!/usr/bin/env python

import os
import csv
import math
import argparse
import matplotlib.pyplot as plt

CSV_HEADER = ['x', 'y', 'z', 'yaw']
MAX_DECEL = 1.0

def draw_waypoints(fname, imgname):
    waypoints = []
    with open(fname) as wfile:
        reader = csv.DictReader(wfile, CSV_HEADER)
        for wp in reader:
            px, py, yaw = float(wp['x']), float(wp['y']), \
                    float(wp['yaw'])/180*math.pi
            waypoints.append((px,py,yaw))
    plt.figure(figsize=(16,12))
    plt.scatter([u[0] for u in waypoints], [u[1] for u in waypoints], s=1)
    yaw_len = 50
    for wp in waypoints[::100]:
        sx, sy = wp[0], wp[1]
        ex, ey = sx+math.cos(wp[2])*yaw_len, sy+math.sin(wp[2])*yaw_len
        plt.plot([sx,ex],[sy,ey], color='red', linewidth=5, alpha=0.5)
    plt.savefig(imgname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read waypoints and plot them.')
    parser.add_argument('csvname', type=str)
    parser.add_argument('imgname', type=str)
    args = parser.parse_args()
    draw_waypoints(args.csvname, args.imgname)
