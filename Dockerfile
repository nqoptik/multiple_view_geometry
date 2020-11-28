FROM ros:kinetic

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/multiple_view_geometry

COPY include include
COPY src src
COPY CMakeLists.txt .

RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make'

RUN rm -rf include
RUN rm -rf src
RUN rm CMakeLists.txt
