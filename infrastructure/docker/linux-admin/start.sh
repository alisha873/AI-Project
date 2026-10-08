#!/bin/bash

set -e

mkdir -p /run/sshd

rsyslogd

exec /usr/sbin/sshd -D -e