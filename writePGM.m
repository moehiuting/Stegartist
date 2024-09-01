function [] = writePGM( image,save_folder,filename)
% image:Ҫд���image����
% save_folder:����ڵ�ǰĿ¼������ļ��У�Ҫ�ȴ������������ᱨ��
% filename:Ҫ������ļ�����
%WRITEPGM �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��

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

