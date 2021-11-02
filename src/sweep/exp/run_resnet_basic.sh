for scale in 1.2 1.4 1.6 1.8 2.0 
do
    echo "python sweep/sweep_CNN_message.py -m resnet -b BasicBlock -s $scale"
    python sweep/sweep_CNN_message.py -m resnet -b BasicBlock -s $scale
done