cd ..
mkdir plugins
cd plugins
git clone https://github.com/ARM-software/SCALE-Sim.git
git clone https://github.com/NVlabs/timeloop

echo "Running and Testing with Scale-Sim"
cd SCALE-Sim/
python scale.py -arch_config=configs/eyeriss.cfg -network=topologies/yolo.csv

cd ../Timeloop/
echo "Running and Testing with Timeloop"
sudo apt-get install scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev libncurses-dev libtinfo-dev libgpm-dev
cd timeloop/src
ln -s ../pat-public/src/pat .
cd ..

scons -j4
source env/setup-env.bash
cd configs/mapper
../../build/timeloop-mapper ./sample.yaml > sample.out