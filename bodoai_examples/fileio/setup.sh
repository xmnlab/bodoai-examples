#!/usr/bin/env bash

# # source: https://docs.bodo.ai/latest/source/aws.html
#
# # Copy your key from your client to all instances.
# # For example, on a Linux clients run this for all instances
# # (find public host names from AWS portal):
# scp -i "user.pem" user.pem ubuntu@ec2-11-111-11-111.us-east-2.compute.amazonaws.com:~/.ssh/id_rsa
#
# # Disable ssh host key check by running this command on all instances:
# echo -e "Host *\n    StrictHostKeyChecking no" > .ssh/config
#
# # Create a host file with list of private hostnames of instances on home directory of all instances:
# echo -e "ip-11-11-11-11.us-east-2.compute.internal\nip-11-11-11-12.us-east-2.compute.internal\n" > hosts
#
# # Set permission for ~/.ssh/config
# chmod 600 ~/.ssh/config
#
# # Set permission for ~/.ssh/id_rsa
# chmod 400 ~/.ssh/id_rsa
