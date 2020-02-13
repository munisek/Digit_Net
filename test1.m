load test1;
dnet=test1;
e=zeros;
for j=1:10
    x=j-1;
    format="%d";
    format=sprintf(format,x);
    leaf="\*.jpg";
    path=strcat(format,leaf);
    imds=imageDatastore(path);
    t=cputime;
    label=classify(dnet,imds);
    e(j)=cputime-t;
    s=numel(label);
    cl=int2str(x);            
    tn=zeros(1,10);
    for i=1:s
        if cl~=label(i)
            tn(1,j)=tn(1,j)+1;
        end
    end
    disp(tn(1,j));
end
