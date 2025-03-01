#ifndef PUBSUB_H
#define PUBSUB_H

#include <ros/ros.h>
#include <string>

template <typename PublishT, typename SubscribeT>
class pubsub{
public:
pubsub(){}
pubsub(std::string publishTopicName, std::string sub1T, std::string sub2T, std::string sub3T, std::string sub4T, std::string sub5T, std::string sub6T, std::string sub7T, std::string sub8T, int queueSize)
{
chatter_pub = nH.advertise<PublishT>(publishTopicName, queueSize);
  sub8 = nH.subscribe<SubscribeT>(sub8T, queueSize, &pubsub::sub8callback,this);
   sub7 = nH.subscribe<SubscribeT>(sub7T,queueSize, &pubsub::sub7callback,this);
  sub1 = nH.subscribe<SubscribeT>(sub1T,queueSize, &pubsub::sub1callback,this);
  sub2 = nH.subscribe<SubscribeT>(sub2T,queueSize,&pubsub::sub2callback,this);
    sub3 = nH.subscribe<SubscribeT>(sub3T,queueSize, &pubsub::sub3callback,this);
   sub4 = nH.subscribe<SubscribeT>(sub4T,queueSize, &pubsub::sub4callback,this);
   sub5 =nH.subscribe<SubscribeT>(sub5T,queueSize, &pubsub::sub5callback,this);
   sub6 = nH.subscribe<SubscribeT>(sub6T,queueSize, &pubsub::sub6callback,this);


//subscriberObject = nH.subscribe<SubscribeT >(subscribeTopicName, queueSize, &pubsub::subcallback,this);
}

void sub1callback(const typename SubscribeT::ConstPtr& recievedMsg);
void sub2callback(const typename SubscribeT::ConstPtr& recievedMsg);
void sub3callback(const typename SubscribeT::ConstPtr& recievedMsg);
void sub4callback(const typename SubscribeT::ConstPtr& recievedMsg);
void sub5callback(const typename SubscribeT::ConstPtr& recievedMsg);
void sub6callback(const typename SubscribeT::ConstPtr& recievedMsg);
void sub7callback(const typename SubscribeT::ConstPtr& recievedMsg);
void sub8callback(const typename SubscribeT::ConstPtr& recievedMsg);
ros::Publisher chatter_pub;

protected: 
ros::Subscriber sub1;
  ros::Subscriber sub2;
  ros::Subscriber sub3;
  ros::Subscriber sub4;
  ros::Subscriber sub5;
  ros::Subscriber sub6;
  ros::Subscriber sub7;
ros::Subscriber sub8;

ros::NodeHandle nH;
};


#endif
