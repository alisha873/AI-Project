FROM ubuntu:24.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        openssh-server \
        rsyslog \
        procps \
        iproute2 && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /run/sshd /var/log && \
    useradd -m -s /bin/bash analyst && \
    echo 'analyst:autoshield_lab' | chpasswd

RUN printf '\nPasswordAuthentication yes\nPermitRootLogin no\n' \
    >> /etc/ssh/sshd_config

COPY start.sh /start.sh

COPY rsyslog/50-autoshield-auth.conf /etc/rsyslog.d/50-autoshield-auth.conf

RUN chmod +x /start.sh

EXPOSE 22

CMD ["/start.sh"]