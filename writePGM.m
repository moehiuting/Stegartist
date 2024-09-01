function [] = writePGM( image,save_folder,filename)
% image:要写入的image对象
% save_folder:相对于当前目录的输出文件夹，要先创建，不创建会报错
% filename:要保存的文件名称
%WRITEPGM 此处显示有关此函数的摘要
%   此处显示详细说明

if ndims(image) == 3
   image = rgb2gray(image);
end

[rows, cols] = size(image);

f = fopen(fullfile(save_folder,filename), 'w');
if f == -1
    error('Could not create file.');
end
fprintf(f, 'P5\n%d\n%d\n255\n', cols, rows);
fwrite(f, image', 'uint8');
fclose(f);
end

