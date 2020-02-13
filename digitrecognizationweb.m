clear;
net=digitnet;
camera=webcam();
while true
    picture=camera.snapshot;
    picture=imresize(picture,[227,227]);
    label=classify(net, picture);
    
    image(picture)
    title(char(label));
    drawnow;
end