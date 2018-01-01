FROM quay.io/pypa/manylinux1_x86_64

ADD install_gtest.sh /install_gtest.sh
RUN /install_gtest.sh && rm /install_gtest.sh
