Img = load('feaSubEImg.mat');
Img1 = Img.class{1,1};
Img2 = Img.class{1,2};
meanI1 = mean(Img1,2);
meanI2 = mean(Img2,2);
Overt = load('feaSubEOvert.mat');
Overt1 = Overt.class{1,1};
Overt2 = Overt.class{1,2};
meanO1 = mean(Overt1,2);
meanO2 = mean(Overt2,2);
distImg = norm(meanI2-meanI1)
distOvert = norm(meanO1-meanO2)