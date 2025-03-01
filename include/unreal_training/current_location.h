#ifndef CURRENT_LOCATION_H
#define CURRENT_LOCATION_H

#include <pid_controller/location.h>
#include <geometry_msgs/Quaternion.h>


extern int test;
extern pid_controller::location current_xyz;
extern pid_controller::location desired_xyz;
extern geometry_msgs::Quaternion inputs;


#pragma once

#endif 
