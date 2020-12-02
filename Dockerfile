# Build stage
FROM ros:kinetic AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/multiple_view_geometry

COPY include include
COPY src src
COPY CMakeLists.txt .

RUN /bin/bash -c '. /opt/ros/kinetic/setup.bash && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make'

# Production stage
FROM ros:kinetic AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/multiple_view_geometry

COPY --from=build /root/multiple_view_geometry/build build

CMD [ "build/structure_from_motion" ]
