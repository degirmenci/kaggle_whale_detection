close all; clear all; clc;
mkdir('imgs_feat_extracted_v8');
mkdir('imgs_feat_extracted_v8/jpg_files');
feat_size_x = 64; feat_size_y = 64;
total_size = 11468;
img_dataset = zeros(feat_size_x, feat_size_y, 3, total_size);
detector_v18p2 = vision.CascadeObjectDetector('detectorFile_v18p2.xml');
detector_v18p3 = vision.CascadeObjectDetector('detectorFile_v18p3.xml');
detector_v8 = vision.CascadeObjectDetector('detectorFile_v8.xml');

% THIS SHOULD START FROM I=0 AND INDEX INSIDE THE LOOP SHOULD CHANGE
% ALSO ADD EXCEPTION FOR 7489 CASE
faulty_inds =[];

for i=0:total_size
    % for i=2
    % for i = randperm(total_size+1)-1
    disp(['Image # ' num2str(i) ' is being processed...']);
    file_name = ['D:\Soysal D\Kaggle Competitions\Whale Recognition\imgs\w_' num2str(i) '.jpg'];
    img_temp = imread(file_name);
    
    
    
    % First try v18p2, try to grab the biggest feature found.
    % If not, try v18p3, try to grab the biggest feature found.
    % If not, grab the whole img.
    disp('Trying v18p2 detector...');
    bbox_temp = step(detector_v18p2, img_temp);
    if numel(bbox_temp)==0
        disp('No features found :(');
        disp('Trying v18p3 detector...');
        bbox_temp = step(detector_v18p3, img_temp);
        if numel(bbox_temp)==0
            disp('No features found :(');
            disp('Trying v8 detector...');
            bbox_temp = step(detector_v8, img_temp);
            if numel(bbox_temp) == 0
                disp('No features found :(');
                disp('Grabbing the whole image...');
                bbox_temp = [1 1 size(img_temp,2) size(img_temp,1)];
            else
                disp('Finding the rectangle that covers all features found...');
                faulty_inds = [faulty_inds, i];
                boundaries_temp = zeros(1,4);
                boundaries_temp(1,1) = min(bbox_temp(:,1));
                boundaries_temp(1,2) = min(bbox_temp(:,2));
                boundaries_temp(1,3) = max(bbox_temp(:,1)+bbox_temp(:,3)-1);
                boundaries_temp(1,4) = max(bbox_temp(:,2)+bbox_temp(:,4)-1);
                bbox_temp = zeros(1,4);
                bbox_temp(1,1) = boundaries_temp(1,1);
                bbox_temp(1,2) = boundaries_temp(1,2);
                bbox_temp(1,3) = boundaries_temp(1,3) - boundaries_temp(1,1) + 1;
                bbox_temp(1,4) = boundaries_temp(1,4) - boundaries_temp(1,2) + 1;
            end
        else
            disp('Success!');
            bbox_sizes = bbox_temp(:,3).*bbox_temp(:,4);
            biggest_feat_ind = find(bbox_sizes == max(bbox_sizes));
            bbox_temp = bbox_temp(biggest_feat_ind,:);
        end
    else
        disp('Success!');
        bbox_sizes = bbox_temp(:,3).*bbox_temp(:,4);
        biggest_feat_ind = find(bbox_sizes == max(bbox_sizes));
        bbox_temp = bbox_temp(biggest_feat_ind,:);
    end
    % Mx4, x, y, width, height (x, y, upper left corner)
    bbox_x_min = bbox_temp(:,1);
    bbox_x_max = bbox_temp(:,1)+bbox_temp(:,3)-1;
    bbox_y_min = bbox_temp(:,2);
    bbox_y_max = bbox_temp(:,2)+bbox_temp(:,4)-1;
    
    %     detected_img_temp = insertObjectAnnotation(img_temp, 'rectangle', bbox_temp, 'FEATURE');
    %     imshow(detected_img_temp);
    
    %     img_cropped = img_temp(bbox_y_min:bbox_y_max,...
    %         bbox_x_min:bbox_x_max,:);
    %     close all; imshow(img_cropped);
    
    
    extra_padding_size_x = round(bbox_temp(:,3)*0.1);
    extra_padding_size_y = round(bbox_temp(:,4)*0.1);
    bbox_x_min = max(bbox_x_min - extra_padding_size_x, 1);
    bbox_x_max = min(bbox_x_max + extra_padding_size_x, size(img_temp,2));
    bbox_y_min = max(bbox_y_min - extra_padding_size_y, 1);
    bbox_y_max = min(bbox_y_max + extra_padding_size_y, size(img_temp,1));
    img_cropped = img_temp(bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max,:);
    img_temp = (rgb2gray(img_cropped));
    img_temp = imresize(img_temp, [64 round(64*size(img_cropped,2)/size(img_cropped,1))]);
    img_temp2 = img_temp;
    mean_img_temp = 109.1472;
    img_temp2(img_temp2>=mean_img_temp) = 255;
    img_temp2(img_temp2<mean_img_temp) = 0;
    
    close all
    imshow(img_temp2);
    [I, J, ~] = find(img_temp2 == 255);
    temp_matr = [I-32, J-round(size(img_temp,2)/2)];
    temp_matr_var = temp_matr'*temp_matr;
    [V, D] = eig(temp_matr_var);
    rotate_angle_deg = atand(V(1,2)/V(1,1));
    %     [Gmag,Gdir] = imgradient(img_temp);
    img_rotated = imrotate(img_cropped, rotate_angle_deg, 'bilinear', 'crop'); imshow(img_rotated);
    %     imshow(img_cropped);
    % %     img_cropped = img_rotated(round(size_x)/2-
    %     % SIFT feats
    %
    % %     figure; imshow(img_cropped);
    % %     close all; imshow(img_cropped);
    %     img_cropped_2 = imresize(img_rotated, [feat_size_x feat_size_y]);
%     file_name_2 = ['D:\Soysal D\Kaggle Competitions\Whale Recognition\imgs_feat_extracted_v8\jpg_files\w_' num2str(i) '.jpg'];
%     imwrite(img_cropped_2, file_name_2);
%     img_dataset(:,:,:,i+1) = img_cropped_2;
end
save('imgs_feat_extracted_v8/img_dataset_v8.mat','-v7.3');
